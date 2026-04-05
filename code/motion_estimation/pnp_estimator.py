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
        
class Keypoint:
    def __init__(self, pt, des, P, frame_count):
        self.pt = pt # 2d euc
        self.des = des
        self.P = P
        self.frame_count = frame_count

class PnPEstimator(MotionEstimator):
    def __init__(self, K, window_size=5):
        super().__init__(K)
        self.essential_matrix_estimator = OpenCVMatrixEstimator(self.K)
        self.landmarks = {}
        self.landmark_id = 0
        self.frame_count = -2
        self.prev_P = None
        self.P = None
        self.keypoints = []
        
        # windowed bundle adjustment vars
        self.BA = BundleAdjuster(K)
        self.window_size = window_size
        self.lm_id_window = deque(maxlen=window_size)
        self.pts2d_window = deque(maxlen=window_size)
        
    def add_to_window(self, pts2d, lm_ids):
        self.pts2d_window.append(pts2d)
        self.lm_id_window.append(lm_ids)
        
    def initial_estimation(self, img, reprojection_tol=2.0):
        if self.frame_count == -2: # first frame
            pose = self.essential_matrix_estimator.estimate(img)
            self.P = self.K @ np.eye(3, 4)
            self.kp = self.essential_matrix_estimator.kp
            self.des = self.essential_matrix_estimator.des
        else:
            pose, _, pts1, pts2 = self.essential_matrix_estimator.estimate(img, return_pts=True)
            self.kp = self.essential_matrix_estimator.kp
            self.des = self.essential_matrix_estimator.des
            kp_kp_matches = self.essential_matrix_estimator.matches
            
            R_wc = pose[:3, :3] # world tocamera
            t_wc = pose[:3, 3]
            R_cw = R_wc.T # camera to world
            t_cw = -R_wc.T @ t_wc
            self.P = self.K @ np.hstack((R_cw, t_cw.reshape(3,1)))

            pts3d_homo = self.triangulate_points(self.prev_P, self.P, pts1, pts2)
            pts3d_euc = pts3d_homo[:3] / pts3d_homo[3]
            
            # depths = pts3d_homo[2] / pts3d_homo[3]
            # depths = depths[depths > 0]
            # scale = np.median(depths)
            # pts3d_homo[:3] /= scale
            # self.prev_P[:, 3] /= scale
            # self.P[:, 3] /= scale
            
            # only take poitns where depth is positive
            pts3d_euc_cam2 = self.P[:, :3] @ pts3d_euc + self.P[:, 3:]
            valid = (pts3d_euc[2] > 0) & (pts3d_euc_cam2[2] > 0)
            valid_idx = np.where(valid)[0]
            pts3d_homo = pts3d_homo[:, valid_idx]
            pts1 = pts1[:, valid_idx]
            pts2 = pts2[:, valid_idx]

            # # DEBUG
            # print("prev_P:\n", self.prev_P)
            # print("curr_P:\n", self.P)
            # print("pts1 shape:", pts1.shape, "sample:", pts1[:, :3])
            # print("pts2 shape:", pts2.shape, "sample:", pts2[:, :3])
            # print("triangulated pts_homo sample:\n", pts_homo[:, :5])

            # depths1 = (self.prev_P @ pts_homo)[2]
            # depths2 = (self.P @ pts_homo)[2]
            # print("depths in prev cam — min:", depths1.min(), "median:", np.median(depths1), "max:", depths1.max())
            # print("depths in curr cam — min:", depths2.min(), "median:", np.median(depths2), "max:", depths2.max())

            # raw_depths = pts_homo[2] / pts_homo[3]
            # print("world z/w — min:", raw_depths.min(), "median:", np.median(raw_depths), "max:", raw_depths.max())
            
            # print("prev_P:\n", self.prev_P)
            # print("curr_P:\n", self.P)
            
            # reproj1 = self.reprojection_error(pts3d_homo, pts1, self.prev_P)
            # reproj2 = self.reprojection_error(pts3d_homo, pts2, self.P)
            # print("reproj err on prev frame - median:", np.median(reproj1), "max:", reproj1.max())
            # print("reproj err on curr frame - median:", np.median(reproj2), "max:", reproj2.max())
            
            matched_kp_indices = [m.trainIdx for m in kp_kp_matches]
            valid_kp_indices = [matched_kp_indices[i] for i in valid_idx]
            des = [self.des[i] for i in valid_kp_indices]

            # add landmarks
            for pt_i, des_i in zip(pts3d_homo.T, des):
                self.landmarks[self.landmark_id] = Landmark(pt_i, des_i)
                self.landmark_id += 1
                
            # form self.matches (3d-2d)
            landmark_ids = list(range(self.landmark_id - len(des), self.landmark_id))
            self.matches = [
                cv2.DMatch(lm_id, valid_kp_indices[i], 0.0)
                for i, lm_id in enumerate(landmark_ids)
            ]
            
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
            
    # def match_landmarks(self, img, prune_threshold=10):
    #     assert prune_threshold >= self.window_size, 'prune_threshold cannot be smaller than window size'
    #     # prune landmarks that havent been matched recently
    #     keep_ids = set()
    #     for id, lm in self.landmarks.items():
    #         if lm.missed_frames < prune_threshold:
    #             keep_ids.add(id)
    #     self.landmarks = {id: lm for id, lm in self.landmarks.items() if id in keep_ids}
                
    #     if len(self.landmarks) < 2:
    #         return []
        
    #     self.kp, self.des = self.tracker.detect(img)

    #     landmark_ids = list(self.landmarks.keys())
    #     landmark_des = np.array([self.landmarks[lm_id].des for lm_id in landmark_ids])
    #     matches = self.tracker.match(landmark_des, self.des)
        
    #     # landmark_pos = np.array([self.landmarks[lm_id].pos for lm_id in landmark_ids])
    #     # for i, m in enumerate(matches[:5]):
    #     #     lm_pos = landmark_pos[m.queryIdx]
    #     #     proj = self.P @ lm_pos
    #     #     proj /= proj[2]
    #     #     print(f"match {i}: lm {m.queryIdx} projects to {proj[:2]}, matched kp {m.trainIdx} at {self.kp[m.trainIdx].pt}")
        
    #     # go from list id to dict id
    #     for m in matches:
    #         m.queryIdx = landmark_ids[m.queryIdx]

    #     matched_ids = set(m.queryIdx for m in matches)
    #     for i, lm in self.landmarks.items():
    #         if i not in matched_ids:
    #             lm.missed_frames += 1
    #         else:
    #             lm.missed_frames = 0 
                
    #     # for m in matches:
    #     #     self.landmarks[m.queryIdx].des = self.des[m.trainIdx]
        
    #     return matches
    
    def get_predicted_P(self):
        if len(self.trajectory) < 2:
            return self.P
        T_prev = self.trajectory[-1]
        T_prev2 = self.trajectory[-2]
        # relative motion between last two frames
        delta = np.linalg.inv(T_prev2) @ T_prev
        T_pred = T_prev @ delta
        R_pred = T_pred[:3, :3].T
        t_pred = -R_pred @ T_pred[:3, 3]
        return self.K @ np.hstack((R_pred, t_pred.reshape(3, 1)))
    
    def match_landmarks(self, img, search_radius=200.0, prune_threshold=11):
        assert prune_threshold >= self.window_size
        self.landmarks = {id: lm for id, lm in self.landmarks.items() 
                        if lm.missed_frames < prune_threshold}
        if len(self.landmarks) < 2:
            return []

        self.kp, self.des = self.tracker.detect(img)
        kp_pts = np.array([kp.pt for kp in self.kp])  # (N, 2)
        
        P_for_search = self.get_predicted_P()

        matches = []
        for lm_id, lm in self.landmarks.items():
            proj = P_for_search @ lm.pos
            proj /= proj[2]
            px, py = proj[0], proj[1]

            # find candidates within search radius
            dists_to_proj = np.linalg.norm(kp_pts - np.array([px, py]), axis=1)
            candidates = np.where(dists_to_proj < search_radius)[0]
            if len(candidates) == 0:
                lm.missed_frames += 1
                continue

            # best descriptor match among candidates
            candidate_des = self.des[candidates]
            des_dists = np.array([cv2.norm(lm.des, candidate_des[i], cv2.NORM_HAMMING)
                                for i in range(len(candidates))])
            best = np.argmin(des_dists)

            if des_dists[best] > 64:
                lm.missed_frames += 1
                continue

            lm.missed_frames = 0
            matches.append(cv2.DMatch(lm_id, int(candidates[best]), float(des_dists[best])))

        return matches
    
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
        
    def get_unmatched_keypoints(self):
        matched_ids = {m.trainIdx for m in self.matches}
        unmatched_ids = [i for i in range(len(self.kp)) if i not in matched_ids]
        if not unmatched_ids:
            return None, None
        unmatched_kp = [self.kp[i] for i in unmatched_ids]
        unmatched_des = [self.des[i] for i in unmatched_ids]
        return unmatched_kp, unmatched_des
    
    def triangulate_ready_tracks(self, query_idx, train_idx, unmatched_kp, unmatched_des, bearing_threshold=np.deg2rad(2), max_new=100, reproj_tol=5.0):
        prev_pts2d_euc = np.array([self.keypoints[i].pt for i in query_idx]).T 
        pts2d_euc = np.array([unmatched_kp[i].pt for i in train_idx]).T
        prev_pts2d_homo = np.vstack([prev_pts2d_euc, np.ones((1, prev_pts2d_euc.shape[1]))])
        pts2d_homo = np.vstack([pts2d_euc, np.ones((1, pts2d_euc.shape[1]))])
        prev_Ps = np.array([self.keypoints[i].P for i in query_idx]) # (N, 4, 4)
        
        angles = bearing_angles(prev_pts2d_homo, pts2d_homo, prev_Ps, self.P, self.K)
        C1s = -np.einsum('nij,nj->ni', prev_Ps[:, :, :3].transpose(0,2,1), prev_Ps[:, :, 3])  # (N, 3)
        C2 = camera_center(self.P, self.K)
        baselines = np.linalg.norm(C1s - C2)
        ready = np.where((angles > bearing_threshold) & (baselines > 1.0))[0]

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

        cross_matched_des = np.array(unmatched_des)[train_idx[ready]]
        
        triangulated_ready_ids = [] # ready frame
        for i in ranked:
            self.landmarks[self.landmark_id] = Landmark(
                pts3d_homo[:, i].copy(),
                cross_matched_des[i].copy()
            )
            self.landmark_id += 1
            triangulated_ready_ids.append(i)
            
        triangulated_ids = query_idx[ready[triangulated_ready_ids]] # unmatched frame
        triangulated_ids_set = set(triangulated_ids.tolist())
        return triangulated_ids_set
        
    def prune_keypoints(self, traingulated_ids, prune_threshold=5):
        prune_ids = set()
        for i, keypoint in enumerate(self.keypoints):
            if i in traingulated_ids: # already triangulated
                prune_ids.add(i)
            elif self.frame_count - keypoint.frame_count > prune_threshold: # stale
                prune_ids.add(i)

        self.keypoints = [keypoint for i, keypoint in enumerate(self.keypoints) if i not in prune_ids]
            
    def add_new_landmarks(self):
        unmatched_kp, unmatched_des = self.get_unmatched_keypoints()
        if unmatched_kp is None:
            return

        # match to stored keypoints
        keypoint_des = np.array([keypoint.des for keypoint in self.keypoints])
        cross_matches = self.tracker.match(keypoint_des, np.array(unmatched_des))
        query_idx = np.array([m.queryIdx for m in cross_matches]) # into self.keypoint
        train_idx = np.array([m.trainIdx for m in cross_matches])  # into self.kp and self.des
        if cross_matches:
            triangulated_ids = self.triangulate_ready_tracks(query_idx, train_idx, unmatched_kp, unmatched_des)
        else:
            triangulated_ids = set()
            
        # remove triangualted and stale keypoints 
        self.prune_keypoints(triangulated_ids)
        
        # add new keypoints
        cross_matched_train_ids = set(train_idx.tolist())
        new_keypoint_ids = [i for i in range(len(unmatched_kp)) if i not in cross_matched_train_ids]

        for i in new_keypoint_ids:
            self.keypoints.append(Keypoint(
                pt=unmatched_kp[i].pt,
                des=unmatched_des[i],
                P=self.P.copy(),
                frame_count=self.frame_count
            ))
    
    def estimate(self, img):
        # initialization for first 2 frames (estimate E to init 3d map)
        if self.frame_count < 0:
            self.initial_estimation(img)
            pose = self.pose_from_P(self.P)
            mask = np.ones(len(self.matches), dtype=bool) if self.frame_count > -2 else np.array([], dtype=bool)
            return pose, mask
        
        self.matches = self.match_landmarks(img)
        print(len(self.landmarks), len(self.matches))
        
        if self.matches is None or len(self.matches) < 6: # not enough matches for DLT
            print("not enough matches — attempting recovery")
            return self.trajectory[-1], np.zeros(len(self.matches), dtype=bool)
        
        pts3d_homo = np.array([self.landmarks[m.queryIdx].pos for m in self.matches]).T
        pts2d_euc = np.array([self.kp[m.trainIdx].pt for m in self.matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1])))) 
        
        # # # --- DEBUG ---
        # P_identity = self.K @ np.eye(3, 4)
        # reproj_per_lm = self.reprojection_error(pts3d_homo, pts2d_homo, P_identity)
        # good = reproj_per_lm < 20
        # bad  = reproj_per_lm >= 20
        # print(f"good matches: {good.sum()}, bad matches: {bad.sum()}")
        # print(f"bad reproj errors: {np.sort(reproj_per_lm[bad])[:10]}")

        # # check one bad landmark
        # bad_idx = np.where(bad)[0][0]
        # bad_m   = self.matches[bad_idx]
        # bad_lm  = self.landmarks[bad_m.trainIdx]
        # proj    = P_identity @ bad_lm.pos
        # proj   /= proj[2]
        # print("bad landmark projects to:", proj[:2])
        # print("bad landmark matched to kp:", self.kp[bad_m.queryIdx].pt)
        # print("bad landmark pos:", bad_lm.pos)
        
        self.add_to_window(pts2d_homo, [m.queryIdx for m in self.matches])
        
        self.P, mask = self._estimate(pts3d_homo, pts2d_homo) 
        pose = self.pose_from_P(self.P)
        
        if len(self.landmarks) > 0:
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
        
        if self.frame_count >= 0:
            # self.bundle_adjust()
            self.add_new_landmarks()
                
        self.frame_count += 1
        self.prev_P = self.P
        self.prev_kp = self.kp
        self.prev_des = self.des
        
        return mask
        
    
class OpenCVPnpEstimator(PnPEstimator):
    def __init__(self, K, window_size=10):
        super().__init__(K, window_size)
        
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
            self.P = self.get_predicted_P()
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