import numpy as np
from collections import deque
import cv2
from motion_estimation import MotionEstimator, EightPointEstimator, OpenCVMatrixEstimator
from bundle_adjuster import BundleAdjuster
from .utils import bearing_angles, camera_center

class Landmark:
    def __init__(self, pos, des):
        self.pos = pos # 4D homo
        self.des = des
        self.missed_frames = 0
        
class CandidateKeypoint:
    def __init__(self, pt, des, P, frame_count):
        self.pt = pt # 2d euc
        # self.des = des
        self.P = P
        self.frame_count = frame_count

class PnPEstimator(MotionEstimator):
    def __init__(self, K, tracker, window_size=5):
        super().__init__(K, tracker)
        self.essential_matrix_estimator = OpenCVMatrixEstimator(self.K, tracker)
        self.tracker = self.essential_matrix_estimator.tracker
        if tracker == 'klt':
            self.tracker.max_corners = 1000
        
        self.frame_count = -2
        self.prev_P = None
        self.P = None
        
        self.landmarks = {}
        self.landmark_id = 0
        
        self.candidate_keypoints = {}
        self.ckp_kp_matches = []
        self.candidate_keypoint_id = 0
        
        # windowed bundle adjustment vars
        self.BA = BundleAdjuster(K)
        self.window_size = window_size
        self.lm_id_window = deque(maxlen=window_size)
        self.pts2d_window = deque(maxlen=window_size)
        
    def add_to_window(self, pts2d, lm_ids):
        self.pts2d_window.append(pts2d)
        self.lm_id_window.append(lm_ids)
        
    def add_landmark(self, pt3d_homo, des):
        self.landmarks[self.landmark_id] = Landmark(pt3d_homo, des)
        self.landmark_id += 1
        
    def add_candidate_keypoint(self, pt):
        keypoint = CandidateKeypoint(pt, None, self.P, self.frame_count)
        self.candidate_keypoints[self.candidate_keypoint_id] = keypoint
        self.candidate_keypoint_id += 1
        
    def initial_estimation(self, img, reprojection_tol=2.0):
        if self.frame_count == -2: # first frame
            pose = self.essential_matrix_estimator.estimate(img)
            self.P = self.K @ np.eye(3, 4)
            self.kp = self.essential_matrix_estimator.kp
            self.des = self.essential_matrix_estimator.des
        else:
            pose, _ = self.essential_matrix_estimator.estimate(img)
            pts1, pts2 = self.essential_matrix_estimator.point_correspondences()
            self.kp = self.essential_matrix_estimator.kp
            self.des = self.essential_matrix_estimator.des
            tracker_matches = self.essential_matrix_estimator.matches

            self.P = self.P_from_pose(pose)
            pts3d_homo = self.triangulate_points(self.prev_P, self.P, pts1, pts2)
            pts3d_euc = pts3d_homo[:3] / pts3d_homo[3]
            
            # depths = pts3d_homo[2] / pts3d_homo[3]
            # depths = depths[depths > 0]
            # scale = np.median(depths)
            # pts3d_homo[:3] /= scale
            # self.prev_P[:, 3] /= scale
            # self.P[:, 3] /= scale
            
            # only take poitns where depth is positive
            pts3d_euc_cam1 = self.prev_P[:3, :3] @ pts3d_euc + self.prev_P[:3, 3:]
            pts3d_euc_cam2 = self.P[:, :3] @ pts3d_euc + self.P[:, 3:]
            valid = (pts3d_euc_cam1[2] > 0) & (pts3d_euc_cam2[2] > 0)
            valid_idx = np.where(valid)[0]
            pts3d_homo = pts3d_homo[:, valid_idx]
            pts1 = pts1[:, valid_idx]
            pts2 = pts2[:, valid_idx]
            
            # reproj1 = self.reprojection_error(pts3d_homo, pts1, self.prev_P)
            # reproj2 = self.reprojection_error(pts3d_homo, pts2, self.P)
            # print("reproj err on prev frame - median:", np.median(reproj1), "max:", reproj1.max())
            # print("reproj err on curr frame - median:", np.median(reproj2), "max:", reproj2.max())
            
            matched_kp_indices = [m.trainIdx for m in tracker_matches]
            valid_kp_indices = [matched_kp_indices[i] for i in valid_idx]
            # des = [self.des[i] for i in valid_kp_indices]

            # add landmarks
            for pt_i in pts3d_homo.T:
                self.add_landmark(pt_i, None)
                
            matched_kp_ids = {m.trainIdx for m in tracker_matches}
            new_kp_ids = [i for i in range(len(self.kp)) if i not in matched_kp_ids]
            for kp_idx in new_kp_ids:
                ckp_idx = self.candidate_keypoint_id
                self.ckp_kp_matches.append(cv2.DMatch(ckp_idx, kp_idx, 0.0))
                self.add_candidate_keypoint(self.kp[kp_idx].pt)
                
            # form self.matches (3d-2d)
            self.matches = [cv2.DMatch(i, valid_kp_indices[i], 0.0) for i in range(self.landmark_id)]
            
        return pose
        
    def pose_from_P(self, P):
        Rt= np.linalg.inv(self.K) @ P
        R = Rt[:, :3]
        t = Rt[:, -1]
        
        U, _, V_t = np.linalg.svd(R)
        R = U @ V_t # project R into space of valid rotation matrices (orthonomal rows)
        
        if np.linalg.det(R) < 0:
            R = -R
            t = -t 
        
        R_wc = R.T
        t_wc = -R.T @ t
        
        pose = np.eye(4)
        pose[:3, :3] = R_wc
        pose[:3, 3] = t_wc
        return pose
    
    def P_from_pose(self, pose):
        R_wc = pose[:3, :3] # world tocamera
        t_wc = pose[:3, 3]
        R_cw = R_wc.T # camera to world
        t_cw = -R_wc.T @ t_wc
        P = self.K @ np.hstack((R_cw, t_cw.reshape(3,1)))
        return P
            
    def match_keypoints(self, img, prune_threshold=10):  
        assert prune_threshold >= self.window_size
        self.landmarks = {lm_idx: lm for lm_idx, lm in self.landmarks.items() if lm.missed_frames < prune_threshold}
           
        if len(self.landmarks) < 2:
            return []
        
        self.kp, tracker_matches = self.tracker.get_matches(img) # tracker_matches is prev_kp <-> kp
        
        updated_lms = set()
        updated_ckps = set()
        kp_to_prev_kp = {m.trainIdx: m.queryIdx for m in tracker_matches}
        prev_kp_to_lm = {m.trainIdx: m.queryIdx for m in self.matches}
        prev_kp_to_ckp = {m.trainIdx: m.queryIdx for m in self.ckp_kp_matches}
        self.ckp_kp_matches = []
        self.matches = []
        new_kp_ids = set()
        
        for kp_idx in range(len(self.kp)):
            prev_kp_idx = kp_to_prev_kp.get(kp_idx)
            
            if prev_kp_idx is not None:
                lm_idx = prev_kp_to_lm.get(prev_kp_idx)
                if lm_idx is not None: # update landmark
                    self.matches.append(cv2.DMatch(lm_idx, kp_idx, 0.0))
                    updated_lms.add(lm_idx)
                else: # update candidate keypoint
                    ckp_idx = prev_kp_to_ckp.get(prev_kp_idx)
                    if ckp_idx is not None:
                        self.ckp_kp_matches.append(cv2.DMatch(ckp_idx, kp_idx, 0.0))
                        updated_ckps.add(ckp_idx)
            else: # save new kp (will add as new ckp's, but need to find P first)
                new_kp_ids.add(kp_idx)
                
        self.candidate_keypoints = {ckp_idx: ckp for ckp_idx, ckp in self.candidate_keypoints.items() if ckp_idx in updated_ckps}
        for lm_idx, lm in self.landmarks.items():
            if lm_idx in updated_lms:
                lm.missed_frames = 0
            else:
                lm.missed_frames += 1
        
        return new_kp_ids
    
    def bundle_adjust(self):
        if self.frame_count < self.window_size - 1:
            return
         
        unique_lm_ids_window = list(set([id for lm_ids in self.lm_id_window for id in lm_ids])) # go back to list to keep order fixed
        pts3d_window = np.array([self.landmarks[id].pos for id in unique_lm_ids_window]).T
        lm_id_to_idx = {lm_id: i for i, lm_id in enumerate(unique_lm_ids_window)}
        correspondences = [[lm_id_to_idx[id] for id in lm_ids] for lm_ids in self.lm_id_window]
        camera_window = np.array(self.trajectory[-self.window_size:])
        
        pts3d_opt, cams_opt = self.BA.adjust(pts3d_window, self.pts2d_window, correspondences, camera_window)
        
        # repalce old treajectory and landamrks
        for i, lm_id in enumerate(unique_lm_ids_window):
            self.landmarks[lm_id].pos = pts3d_opt[:, i]
        self.trajectory[-self.window_size:] = list(cams_opt)
        self.P = self.P_from_pose(cams_opt[-1])
    
    def triangulate_ready_tracks(self, bearing_threshold=np.deg2rad(10), max_new=500, reproj_tol=5.0):
        ckp_ids = [m.queryIdx for m in self.ckp_kp_matches]
        kp_ids = [m.trainIdx for m in self.ckp_kp_matches]
        
        prev_pts2d_euc = np.array([self.candidate_keypoints[i].pt for i in ckp_ids]).T
        pts2d_euc = np.array([self.kp[i].pt for i in kp_ids]).T
        prev_pts2d_homo = np.vstack([prev_pts2d_euc, np.ones((1, prev_pts2d_euc.shape[1]))])
        pts2d_homo = np.vstack([pts2d_euc, np.ones((1, pts2d_euc.shape[1]))])
        prev_Ps = np.array([self.candidate_keypoints[i].P for i in ckp_ids]) # (N, 4, 4)
        
        angles = bearing_angles(prev_pts2d_homo, pts2d_homo, prev_Ps, self.P, self.K)
        C1s = -np.einsum('nij,nj->ni', prev_Ps[:, :, :3].transpose(0,2,1), prev_Ps[:, :, 3])  # (N, 3)
        C2 = camera_center(self.P, self.K)
        baselines = np.linalg.norm(C1s - C2[np.newaxis, :], axis=1)
        # ready = np.where((angles > bearing_threshold) & (baselines > 0.5))[0]
        ready = np.where(angles > bearing_threshold)[0]

        prev_pts2d_homo = prev_pts2d_homo[:, ready]
        pts2d_homo = pts2d_homo[:, ready]
        prev_Ps = prev_Ps[ready]
        pts3d_homo = self.triangulate_points_batch(prev_Ps, self.P, prev_pts2d_homo, pts2d_homo)

        # validity checks
        pts3d_euc = pts3d_homo[:3] / pts3d_homo[3]
        pts3d_euc_cam1 = np.einsum('nij,jn->in', prev_Ps[:, :, :3], pts3d_euc) + prev_Ps[:, :, 3].T
        pts3d_euc_cam2 = self.P[:3, :3] @ pts3d_euc + self.P[:3, 3:]
        valid = (pts3d_euc_cam1[2] > 0) & (pts3d_euc_cam2[2] > 0)
        reproj_err = self.reprojection_error(pts3d_homo, pts2d_homo, self.P)
        valid &= reproj_err < reproj_tol
        # depths = pts3d_homo[2] / pts3d_homo[3]
        # valid &= (depths > 1.0) & (depths < 200.0)
        valid_idx = np.where(valid)[0]

        ranked = sorted(valid_idx, key=lambda i: reproj_err[i])[:max_new]
        ready_kp_ids = np.array(kp_ids)[ready] 
        ready_ckp_ids = np.array(ckp_ids)[ready]
        triangulated_ckp_ids = set(ready_ckp_ids[ranked].tolist())
        
        for i in ranked:
            self.matches.append(cv2.DMatch(self.landmark_id, ready_kp_ids[i], 0.0))
            self.add_landmark(pts3d_homo[:, i].copy(), None)
        
        return triangulated_ckp_ids
            
    def add_new_landmarks(self, new_kp_ids):
        triangulated_ckp_ids = self.triangulate_ready_tracks()
        print('new', len(self.candidate_keypoints), len(triangulated_ckp_ids))
        
        # remove triangualted keypoints 
        self.candidate_keypoints = {ckp_idx: ckp for ckp_idx, ckp in self.candidate_keypoints.items() if not ckp_idx in triangulated_ckp_ids}
        ckp_kp_matches = []
        for m in self.ckp_kp_matches:
            ckp_idx = m.queryIdx
            kp_idx = m.trainIdx
            if ckp_idx not in triangulated_ckp_ids:
                ckp_kp_matches.append(cv2.DMatch(ckp_idx, kp_idx, 0.0))
        self.ckp_kp_matches = ckp_kp_matches
        
        # add new keypoints
        for kp_idx in new_kp_ids:
            ckp_idx = self.candidate_keypoint_id # save this BEFORE adding (bc it gets icnremented in func)
            self.add_candidate_keypoint(self.kp[kp_idx].pt)
            self.ckp_kp_matches.append(cv2.DMatch(ckp_idx, kp_idx, 0.0))

    
    def estimate(self, img):
        # initialization for first 2 frames (estimate E to init 3d map)
        if self.frame_count < 0:
            self.initial_estimation(img)
            pose = self.pose_from_P(self.P)
            mask = np.ones(len(self.matches), dtype=bool) if self.frame_count > -2 else np.array([], dtype=bool)
            return pose, mask
        
        new_kp_ids = self.match_keypoints(img)
        print(len(self.landmarks), len(self.matches))
        print()
        
        if self.matches is None or len(self.matches) < 6: # not enough matches for DLT
            print("not enough matches — attempting recovery")
            return self.trajectory[-1], np.zeros(len(self.matches), dtype=bool)
        
        pts3d_homo = np.array([self.landmarks[m.queryIdx].pos for m in self.matches]).T
        pts2d_euc = np.array([self.kp[m.trainIdx].pt for m in self.matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1])))) 
        
        self.add_to_window(pts2d_homo, [m.queryIdx for m in self.matches])
        
        self.P, mask = self._estimate(pts3d_homo, pts2d_homo) 
        
        if self.frame_count >= 0:
            # self.bundle_adjust()
            self.add_new_landmarks(new_kp_ids)
            
        pose = self.pose_from_P(self.P)
        
        # if len(self.landmarks) > 0:
        all_depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks.values()])
        all_depths = all_depths[all_depths > 0]
        translation = np.linalg.norm(pose[:3, 3])
        print(f"frame={self.frame_count} | map_median_depth={np.median(all_depths):.2f} | cam_pos={pose[:3,3].round(2)} | translation={translation:.2f}")
            
        return pose, mask
    
    def _estimate(self, pts3d_homo, pts2d_homo):
        raise NotImplementedError
    
    def step(self, img):
        pose, mask = self.estimate(img)
        self.trajectory.append(pose)
                
        self.frame_count += 1
        self.prev_P = self.P
        self.prev_kp = self.kp
        self.prev_des = self.des
        
        return mask
        
    
class OpenCVPnpEstimator(PnPEstimator):
    def __init__(self, K, tracker, window_size=10):
        super().__init__(K, tracker, window_size)
        
    def _estimate(self, pts3d_homo, pts2d_homo):
        pts3d_euc = (pts3d_homo / pts3d_homo[3])[:3].T
        pts2d_euc = (pts2d_homo / pts2d_homo[2])[:2].T
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_euc.astype(np.float32),
            pts2d_euc.astype(np.float32),
            self.K,
            np.zeros((4, 1)),
            reprojectionError=5.0,
            confidence=0.9,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP  
        )
        print("matches:", len(self.matches))
        print("success:", success)
        print("inliers:", len(inliers) if inliers is not None else 0)
        
        if not success:
            print('solver failed')
            # self.P = self.get_predicted_P()
            return self.P, np.zeros(len(self.matches), dtype=bool)
        
        if success and inliers is not None and len(inliers) >= 6:
            inlier_idx = inliers.flatten()
            _, rvec, tvec = cv2.solvePnP(
                pts3d_euc[inlier_idx].astype(np.float32),
                pts2d_euc[inlier_idx].astype(np.float32),
                self.K,
                np.zeros((4, 1)),
                rvec, tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        P = self.K @ np.hstack((R, t.reshape(3,1))) 
        
        mask = np.zeros(len(self.matches), dtype=bool)
        if inliers is not None:
            mask[inliers.flatten()] = True
            
        return P, mask