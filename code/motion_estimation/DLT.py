import numpy as np
from motion_estimation import MotionEstimator, EightPointEstimator
import cv2
from collections import deque

MAX_LANDMARKS = 5000

class Landmark:
    def __init__(self, pos, des):
        self.pos = pos # 4D homo
        self.des = des
        self.missed_frames = 0

class DLTEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        self.eight_point_estimator = EightPointEstimator(self.K)
        self.returns_global = True
        self.landmarks = deque(maxlen=MAX_LANDMARKS)
        self.frame_count = 0
        self.prev_P = None
        self.P = None
        self.last_pose = np.eye(4)
        self.frame_count = 0
        self.keyframe_P = None
        self.keyframe_kp = None
        self.keyframe_des = None
        
    def initial_estimation(self, img, reprojection_tol=2.0):
        if self.frame_count == 0: # first frame
            pose = self.eight_point_estimator.estimate(img)
            self.P = self.K @ np.eye(3, 4)
            
            self.keyframe_P = self.P
            self.keyframe_kp = self.eight_point_estimator.kp
            self.keyframe_des = self.eight_point_estimator.des
        else:
            rel_motion, pts1, pts2, des = self.eight_point_estimator.estimate(img, return_pts_des=True)
            pose = self.last_pose @ rel_motion
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
            
            reproj1 = self.reprojection_error(pts_homo, pts1, self.prev_P)
            reproj2 = self.reprojection_error(pts_homo, pts2, self.P)
            # valid &= (reproj1 < reprojection_tol) & (reproj2 < reprojection_tol)

            valid_idx = np.where(valid)[0]
            pts_homo = pts_homo[:, valid_idx]
            des = [des[i] for i in valid_idx]
            
            depths = pts_homo[2] / pts_homo[3]          # z/w
            median_depth = np.median(depths[(depths > 0.1) & (depths < 10)])
            pts_homo[:3] /= median_depth   
            
            print("reproj err on prev frame - median:", np.median(reproj1), "max:", reproj1.max())
            print("reproj err on curr frame - median:", np.median(reproj2), "max:", reproj2.max())

            # add landmarks
            for pt_i, des_i in zip(pts_homo.T, des):
                self.landmarks.append(Landmark(pt_i, des_i))
                           
        self.matches = self.eight_point_estimator.matches
        return pose
        
    def dlt(self, pts3d, pts2d):
        assert pts3d.shape[-1] == pts2d.shape[-1], "point correspondecnes not of equal length"
        N = pts3d.shape[-1]
        assert N >= 6, "need at least 6 points for DLT"
        
        # normalize
        T1 = self.normalize(pts3d)
        T2 = self.normalize(pts2d)
        pts3d_norm = T1 @ pts3d
        pts2d_norm = T2 @ pts2d
        
        A = np.zeros((2 * N, 12))
        for i in range(N):
            X = pts3d_norm[:, i]
            x, y, w = pts2d_norm[:, i]
            
            
            A[2*i] = np.hstack((np.zeros(4), -w*X, y*X))
            A[2*i + 1] = np.hstack((w*X, np.zeros(4), -x*X))
            
        _, _, V_t = np.linalg.svd(A)
        P = V_t[-1].reshape(3, 4)
        P /= P[-1, -1] # set w=1
        
        # unnormalzie
        P = np.linalg.inv(T2) @ P @ T1 
        return P
    
    def reprojection_error(self, pts3d, pts2d, P):
        proj = P @ pts3d
        proj /= proj[2, :] # normalize

        error = np.linalg.norm(proj[:2] - pts2d[:2], axis=0)
        return error
    
    def dlt_ransac(self, pts3d, pts2d, tol=3.0, max_iterations=1000, min_inliers=0.75):
        N = pts3d.shape[1]
        assert N >= 6, "need at least 6 points for DLT"
        
        max_inliers = -1
        best_inliers_mask = None
        for i in range(max_iterations):
            sample_ids = np.random.choice(N, 6, replace=False)
            pts3d_sample = pts3d[:, sample_ids]
            pts2d_sample = pts2d[:, sample_ids]
            
            P = self.dlt(pts3d_sample, pts2d_sample)
            
            error = self.reprojection_error(pts3d, pts2d, P)
            inliers_mask = error < tol
            n_inliers = inliers_mask.sum()
            
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inliers_mask = inliers_mask
                
            if n_inliers / N >= min_inliers:
                break
            
        inlier_pts3d = pts3d[:, best_inliers_mask]
        inlier_pts2d = pts2d[:, best_inliers_mask]
        P = self.dlt(inlier_pts3d, inlier_pts2d)
        
        return P
    
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
        c_kf = self.camera_center(self.keyframe_P)
        c_curr = self.camera_center(self.P)
        baseline = np.linalg.norm(c_curr - c_kf)
        depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks])
        median_depth = np.median(depths[depths > 0])
        return baseline / median_depth > 0.20 
    
    def add_new_landmarks(self, max_new=100, reprojection_tol=2.0):
        c_kf = self.camera_center(self.keyframe_P)
        c_curr = self.camera_center(self.P)
        baseline = np.linalg.norm(c_curr - c_kf)
        
        # scale baseline relative to median landmark depth
        depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] 
                            for lm in self.landmarks])
        depths = depths[depths > 0]
        if len(depths) == 0:
            return
        
        median_depth = np.median(depths)
        relative_baseline = baseline / median_depth
        
        if relative_baseline < 0.02:  # less than 2% of scene depth — pure rotation
            return

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
        cross_matches = self.tracker.match(np.array(unmatched_des), np.array(self.keyframe_des))
        if not cross_matches:
            return

        # traingualte 3d points
        pts2d_euc = np.array([unmatched_kp[m.queryIdx].pt for m in cross_matches]).T
        prev_pts2d_euc = np.array([self.keyframe_kp[m.trainIdx].pt for m in cross_matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1]))))
        prev_pts2d_homo = np.vstack((prev_pts2d_euc, np.ones((1, prev_pts2d_euc.shape[1]))))
        des_new = [unmatched_des[m.queryIdx] for m in cross_matches]
        pts3d_homo = self.triangulate_points(self.keyframe_P, self.P, prev_pts2d_homo, pts2d_homo)

        # filter out landamrks behind camera or with bad reprojection
        valid  = ((self.keyframe_P @ pts3d_homo)[2] > 0) & ((self.P @ pts3d_homo)[2] > 0)
        reproj_err = self.reprojection_error(pts3d_homo, pts2d_homo, self.P)
        prev_reproj_err = self.reprojection_error(pts3d_homo, prev_pts2d_homo, self.keyframe_P)
        valid &= (reproj_err < reprojection_tol) & (prev_reproj_err < reprojection_tol)
        new_depths = pts3d_homo[2] / pts3d_homo[3]
        depth_median = np.median(new_depths[valid])
        valid &= (new_depths > depth_median * 0.1) & (new_depths < depth_median * 5)
        valid_idx = np.where(valid)[0]
        
        # print("reproj err on prev frame - median:", np.median(prev_reproj_err), "max:", prev_reproj_err.max())
        # print("reproj err on curr frame - median:", np.median(reproj_err), "max:", reproj_err.max())

        # add points with lowest reprojection error
        ranked = sorted(valid_idx, key=lambda i: reproj_err[i])[:max_new]
        
        for i in ranked:
            new_lm = Landmark(pts3d_homo[:, i].copy(), des_new[i].copy())
            self.landmarks.append(new_lm)
                
    def match_landmarks(self, img, prune_threshold=15, max_proj_dist=100):
        # prune landmarks that havent been matched recently
        new_dq = deque(maxlen=MAX_LANDMARKS)
        for lm in self.landmarks:
            if lm.missed_frames < prune_threshold:
                new_dq.append(lm)
        self.landmarks = new_dq
        if len(self.landmarks) < 2:
            return []
        
        self.kp, self.des = self.tracker.detect(img)
        h, w = img.shape[:2]
    
        matches = []

        # only add matches that are near expected projection
        matches = self.tracker.match(self.des, np.array([lm.des for lm in self.landmarks]))
        # for m in raw_matches:
        #     # lm = self.landmarks[m.trainIdx]
        #     # proj = self.prev_P @ lm.pos
        #     # if proj[2] <= 0:
        #     #     continue
        #     # proj /= proj[2]
        #     # px, py = proj[:2]
        #     # # if not (0 <= px < w and 0 <= py < h):
        #     # #     continue
        #     # kp_pt = np.array(self.kp[m.queryIdx].pt)
        #     # if np.linalg.norm(kp_pt - np.array([px, py])) > max_proj_dist:
        #     #     continue
        #     matches.append(m)

        matched_ids = set(m.trainIdx for m in matches)
        for i, lm in enumerate(self.landmarks):
            if i not in matched_ids:
                lm.missed_frames += 1
                
        # for m in matches:
        #     self.landmarks[m.trainIdx].des = self.des[m.queryIdx]
                
        return matches
        
    def estimate(self, img, ransac=True):
        self.prev_P = self.P
        self.prev_kp = self.eight_point_estimator.kp
        self.prev_des = self.eight_point_estimator.des
        
        # initialization for first 2 frames (estimate E to init 3d map)
        if self.frame_count < 2:
            pose = self.initial_estimation(img)
            self.frame_count += 1
            self.last_pose = pose
            return pose
        
        self.matches = self.match_landmarks(img)
        print(len(self.landmarks), len(self.matches))
        
        
        if self.matches is None or len(self.matches) < 6: # not enough matches for DLT
            print("not enough matches — attempting recovery")
            # reset keyframe to force fresh landmark addition next frame
            self.keyframe_P = self.prev_P
            self.keyframe_kp = self.prev_kp
            self.keyframe_des = self.prev_des
            return self.last_pose
        
        pts3d_homo = np.array([self.landmarks[m.trainIdx].pos for m in self.matches]).T
        pts2d_euc = np.array([self.kp[m.queryIdx].pt for m in self.matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1])))) 
        
        # opencv
        if True:
            pts3d = np.array([self.landmarks[m.trainIdx].pos[:3] / self.landmarks[m.trainIdx].pos[3] for m in self.matches])
            pts2d = np.array([self.kp[m.queryIdx].pt for m in self.matches])
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d.astype(np.float32),
                pts2d.astype(np.float32),
                self.K,
                np.zeros((4, 1)),
                reprojectionError=8.0,
                confidence=0.75,
                iterationsCount=1000,
                flags=cv2.SOLVEPNP_EPNP
            )
            # print("matches:", len(self.matches))
            # print("success:", success)
            # print("inliers:", len(inliers) if inliers is not None else 0)
            
            if not success:
                print('solver failed')
                self.prev_P = self.P
                self.prev_kp  = self.kp
                self.prev_des = self.des
                return self.last_pose
            
            if success and inliers is not None and len(inliers) >= 6:
                inlier_idx = inliers.flatten()
                success2, rvec, tvec = cv2.solvePnP(
                    pts3d[inlier_idx].astype(np.float32),
                    pts2d[inlier_idx].astype(np.float32),
                    self.K,
                    np.zeros((4, 1)),
                    rvec, tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            R_wc = R.T
            t_wc = -R.T @ t

            pose = np.eye(4)
            pose[:3, :3] = R_wc
            pose[:3, 3] = t_wc

            self.P = self.K @ np.hstack((R, t.reshape(3,1))) 
            
        # if ransac:
        #     self.P = self.dlt_ransac(pts3d_homo, pts2d_homo)

        # # else:
        # #     self.P = self.dlt(pts3d_homo, pts2d_homo)        
        # pose = self.pose_from_P(self.P)
        
        if len(self.landmarks) > 0:
            all_depths = np.array([(self.P @ lm.pos)[2] / lm.pos[3] for lm in self.landmarks])
            all_depths = all_depths[all_depths > 0]
            translation = np.linalg.norm(pose[:3, 3])
            print(f"frame={self.frame_count} | map_median_depth={np.median(all_depths):.2f} | cam_pos={pose[:3,3].round(2)} | translation={translation:.2f}")
        
        if not np.isfinite(self.P).all():
            print('not finite')
            self.prev_P = self.P if self.P is not None else self.prev_P
            self.prev_kp = self.kp
            self.prev_des = self.des
            return self.last_pose

        
        self.add_new_landmarks()
        
        if self.should_add_keyframe():
            self.keyframe_P = self.P
            self.keyframe_kp = self.kp
            self.keyframe_des = self.des
            
        self.prev_P = self.P
        self.prev_kp = self.kp
        self.prev_des = self.des
        self.last_pose = pose 
        self.frame_count += 1
        return pose
        