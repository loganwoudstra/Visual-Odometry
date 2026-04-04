import numpy as np
from collections import deque
import cv2
from motion_estimation import MotionEstimator, EightPointEstimator
from bundle_adjuster import BundleAdjuster

class Landmark:
    def __init__(self, pos, des):
        self.pos = pos # 4D homo
        self.des = des
        self.missed_frames = 0
        
class KeyFrame:
    def __init__(self, P, kp, des, frame_count) -> None:
        self.P = P
        self.kp = kp
        self.des = des
        self.frame_count = frame_count

class PnPEstimator(MotionEstimator):
    def __init__(self, K, window_size=5):
        super().__init__(K)
        self.eight_point_estimator = EightPointEstimator(self.K)
        self.landmarks = {}
        self.landmark_id = 0
        self.frame_count = -2
        self.prev_P = None
        self.P = None
        self.keyframe = KeyFrame(None, None, None, None)
        
        # windowed bundle adjustment vars
        self.BA = BundleAdjuster(K)
        self.window_size = window_size
        self.lm_id_window = deque(maxlen=window_size)
        self.pts2d_window = deque(maxlen=window_size)
        
    def add_to_window(self, pts2d, lm_ids):
        self.pts2d_window.append(pts2d)
        self.lm_id_window.append(lm_ids)
        
    def reprojection_error(self, pts3d, pts2d, P):
        proj = P @ pts3d
        proj /= proj[2, :] # normalize

        error = np.linalg.norm(proj[:2] - pts2d[:2], axis=0)
        return error
        
    def initial_estimation(self, img, reprojection_tol=2.0):
        if self.frame_count == -2: # first frame
            pose = self.eight_point_estimator.estimate(img)
            self.P = self.K @ np.eye(3, 4)
            self.kp = self.eight_point_estimator.kp
            self.des = self.eight_point_estimator.des
            self.keyframe = KeyFrame(self.P, self.kp, self.des, self.frame_count)

        else:
            pose, pts1, pts2 = self.eight_point_estimator.estimate(img, return_pts=True)
            self.kp = self.eight_point_estimator.kp
            self.des = self.eight_point_estimator.des
            self.matches = self.eight_point_estimator.matches
            
            R_wc = pose[:3, :3] # world tocamera
            t_wc = pose[:3, 3]
            R_cw = R_wc.T # camera to world
            t_cw = -R_wc.T @ t_wc
            self.P = self.K @ np.hstack((R_cw, t_cw.reshape(3,1)))

            pts_homo = self.triangulate_points(self.prev_P, self.P, pts1, pts2)
            
            # only take poitns where depth is positive and reprojection error is sufficently small
            proj1 = self.prev_P @ pts_homo
            proj2 = self.P @ pts_homo
            valid = (proj1[2] > 0) & (proj2[2] > 0)

            valid_idx = np.where(valid)[0]
            pts_homo = pts_homo[:, valid_idx]
            pts1 = pts1[:, valid_idx]
            pts2 = pts2[:, valid_idx]
            # des = [self.des[i] for i in valid_idx]
            matched_kp_indices = [m.trainIdx for m in self.matches]  # or queryIdx depending on convention
            des = [self.des[matched_kp_indices[i]] for i in valid_idx]
            
            depths = pts_homo[2] / pts_homo[3]
            scale = np.median(depths[depths > 0.1])
            # scale = np.median(depths[(depths > 0.1)])
            pts_homo[:3] /= scale
            self.P[:, 3] /= scale      # translation scales with points
            
            reproj1 = self.reprojection_error(pts_homo, pts1, self.prev_P)
            reproj2 = self.reprojection_error(pts_homo, pts2, self.P)

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
            
            print("reproj err on prev frame - median:", np.median(reproj1), "max:", reproj1.max())
            print("reproj err on curr frame - median:", np.median(reproj2), "max:", reproj2.max())

            # add landmarks
            for pt_i, des_i in zip(pts_homo.T, des):
                self.landmarks[self.landmark_id] = Landmark(pt_i, des_i)
                self.landmark_id += 1
                
            # window
            # added_lm_ids = [i for i in range(self.landmark_id)] # all landmarks are seen in frame 0 and 1
            # self.add_to_window(pts1, added_lm_ids)
            # self.add_to_window(pts2, added_lm_ids)
                           
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
    
    def camera_center(self, P):
        M = np.linalg.inv(self.K) @ P  # 3x4 [R|t]
        R = M[:, :3]
        t = M[:, 3]
        return -R.T @ t  # 3D world position of camera
    
    def should_add_keyframe(self):
        # return True
        c_kf = self.camera_center(self.keyframe.P)
        c_curr = self.camera_center(self.P)
        baseline = np.linalg.norm(c_curr - c_kf)
        depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks.values()])
        median_depth = np.median(depths[depths > 0])
        return baseline / median_depth > 0.35 
    
    def add_new_landmarks(self, max_new=200, reprojection_tol=5.0):
        # c_kf = self.camera_center(self.keyframe.P)
        # c_curr = self.camera_center(self.P)
        # baseline = np.linalg.norm(c_curr - c_kf)
        
        # # scale baseline relative to median landmark depth
        # depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks.values()])
        # depths = depths[depths > 0]
        # if len(depths) == 0:
        #     return
        
        # median_depth = np.median(depths)
        # relative_baseline = baseline / median_depth
        
        # if relative_baseline < 0.05:  # less than 2% of scene depth — pure rotation
        #     return

        # get features not matched with landmarks
        matched_ids = set()
        for m in self.matches:
            matched_ids.add(m.queryIdx)
        unmatched_ids = [i for i in range(len(self.kp)) if i not in matched_ids]
        if not unmatched_ids:
            return
        unmatched_kp = [self.kp[i] for i in unmatched_ids]
        unmatched_des = [self.des[i] for i in unmatched_ids]

        # find matches (of unmatched) between curr and prev frame
        cross_matches = self.tracker.match(np.array(unmatched_des), np.array(self.keyframe.des))
        if not cross_matches:
            return

        # traingualte 3d points
        pts2d_euc = np.array([unmatched_kp[m.queryIdx].pt for m in cross_matches]).T
        prev_pts2d_euc = np.array([self.keyframe.kp[m.trainIdx].pt for m in cross_matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1]))))
        prev_pts2d_homo = np.vstack((prev_pts2d_euc, np.ones((1, prev_pts2d_euc.shape[1]))))
        des_new = [unmatched_des[m.queryIdx] for m in cross_matches]
        pts3d_homo = self.triangulate_points(self.keyframe.P, self.P, prev_pts2d_homo, pts2d_homo)

        # filter out landamrks behind camera or with bad reprojection
        valid  = ((self.keyframe.P @ pts3d_homo)[2] > 0) & ((self.P @ pts3d_homo)[2] > 0)
        reproj_err = self.reprojection_error(pts3d_homo, pts2d_homo, self.P)
        # prev_reproj_err = self.reprojection_error(pts3d_homo, prev_pts2d_homo, self.keyframe.P)
        # valid &= (reproj_err < reprojection_tol) & (prev_reproj_err < reprojection_tol)
        # new_depths = pts3d_homo[2] / pts3d_homo[3]
        # depth_median = np.median(new_depths[valid])
        # valid &= (new_depths > depth_median * 0.1) & (new_depths < depth_median * 1.5)
        valid_idx = np.where(valid)[0]
        
        # valid_idx = valid_idx[reproj_err[valid_idx] < reprojection_tol]
        # if len(valid_idx) == 0:
        #     return

        # # Scale new points to match existing map
        # existing_depths = np.array([
        #     (self.P @ lm.pos)[2] / lm.pos[3]
        #     for lm in self.landmarks.values()
        # ])
        # existing_depths = existing_depths[existing_depths > 0]

        # new_depths = pts3d_homo[2, valid_idx] / pts3d_homo[3, valid_idx]
        # new_depths = new_depths[new_depths > 0]

        # if len(existing_depths) > 5 and len(new_depths) > 5:
        #     scale = np.median(existing_depths) / np.median(new_depths)
        #     # sanity gate — if scale correction is huge, triangulation is degenerate
        #     if 0.1 < scale < 10.0:
        #         pts3d_homo[:3, :] *= scale  # only scale XYZ, not W
        #     else:
        #         return  # skip adding these landmarks entirely
        
        # print("reproj err on prev frame - median:", np.median(prev_reproj_err), "max:", prev_reproj_err.max())
        # print("reproj err on curr frame - median:", np.median(reproj_err), "max:", reproj_err.max())

        # add points with lowest reprojection error
        ranked = sorted(valid_idx, key=lambda i: reproj_err[i])[:max_new]
        
        for i in ranked:
            self.landmarks[self.landmark_id] = Landmark(pts3d_homo[:, i].copy(), des_new[i].copy())
            self.landmark_id += 1
            # TODO: add points to window?
                
    def match_landmarks(self, img, prune_threshold=25):
        assert prune_threshold >= self.window_size, 'prune_threshold cannot be larger than window size'
        # prune landmarks that havent been matched recently
        keep_ids = set()
        for id, lm in self.landmarks.items():
            if lm.missed_frames < prune_threshold:
                keep_ids.add(id)
        self.landmarks = {id: lm for id, lm in self.landmarks.items() if id in keep_ids}
                
        if len(self.landmarks) < 2:
            return []
        
        self.kp, self.des = self.tracker.detect(img)
        h, w = img.shape[:2]

        landmark_ids = list(self.landmarks.keys())
        landmark_des = np.array([self.landmarks[lm_id].des for lm_id in landmark_ids])
        matches = self.tracker.match(self.des, landmark_des)
        
        # go from list id to dict id
        for m in matches:
            m.trainIdx = landmark_ids[m.trainIdx]

        matched_ids = set(m.trainIdx for m in matches)
        for i, lm in self.landmarks.items():
            if i not in matched_ids:
                lm.missed_frames += 1
            else:
                lm.missed_frames = 0 
                
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
    
    def estimate(self, img):
        # initialization for first 2 frames (estimate E to init 3d map)
        if self.frame_count < 0:
            self.initial_estimation(img)
            pose = self.pose_from_P(self.P)
            return pose
        
        self.matches = self.match_landmarks(img)
        print(len(self.landmarks), len(self.matches))
        
        if self.matches is None or len(self.matches) < 6: # not enough matches for DLT
            print("not enough matches — attempting recovery")
            # reset keyframe to force fresh landmark addition next frame
            self.keyframe = KeyFrame(self.prev_P, self.prev_kp, self.prev_des, self.frame_count-1)
            return self.trajectory[-1]
        
        pts3d_homo = np.array([self.landmarks[m.trainIdx].pos for m in self.matches]).T
        pts2d_euc = np.array([self.kp[m.queryIdx].pt for m in self.matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1])))) 
        
        self.add_to_window(pts2d_homo, [m.trainIdx for m in self.matches])
        
        self.P = self._estimate(pts3d_homo, pts2d_homo) 
        pose = self.pose_from_P(self.P)
        
        if len(self.landmarks) > 0:
            all_depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks.values()])
            all_depths = all_depths[all_depths > 0]
            translation = np.linalg.norm(pose[:3, 3])
            print(f"frame={self.frame_count} | map_median_depth={np.median(all_depths):.2f} | cam_pos={pose[:3,3].round(2)} | translation={translation:.2f}")
            
        return pose
    
    def _estimate(self, pts3d_homo, pts2d_homo):
        raise NotImplementedError
    
    def step(self, img):
        pose = self.estimate(img)
        self.trajectory.append(pose)
        
        if self.frame_count >= 0:
            # self.bundle_adjust()
            self.add_new_landmarks()
            if self.should_add_keyframe():
                print('Keyframe: ', self.frame_count)
                self.keyframe = KeyFrame(self.P, self.kp, self.des, self.frame_count)
                
        self.frame_count += 1
        self.prev_P = self.P
        self.prev_kp = self.kp
        self.prev_des = self.des
        
    
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
            confidence=0.75,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP
        )
        # print("matches:", len(self.matches))
        # print("success:", success)
        # print("inliers:", len(inliers) if inliers is not None else 0)
        
        if not success:
            print('solver failed')
            return self.P
        
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
        return P