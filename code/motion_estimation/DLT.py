from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
from motion_estimation import MotionEstimator, EightPointEstimator

class Landmark:
    def __init__(self, pos, des):
        self.pos = pos # 4D homo
        self.des = des

class DLTEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        self.eight_point_estimator = EightPointEstimator(self.K)
        self.returns_global = True
        self.landmarks = []
        self.initialized = -1 # -1=processed no frames, 0=processed frame 1, 1=proccseed frames 1 and 2 (ie. initialized)
        self.prev_P = None
        self.prev_kp = None
        self.prev_des = None
        
    def initial_estimation(self, img):
        if self.initialized == -1: # first frame
            pose = self.eight_point_estimator.estimate(img)
        elif self.initialized == 0: # second frame
            pose, pts1, pts2, descriptions = self.eight_point_estimator.estimate(img, return_pts_des=True)
            P1 = np.eye(4)
            P2 = self.K @ pose[:3]
            pts_homo = self.triangulate_points(P1, P2, pts1, pts2)
            
            for pt, des in zip(pts_homo.T, descriptions):
                self.landmarks.append(Landmark(pt, des))
                
            self.prev_P = P2
            self.prev_kp, self.prev_des = self.eight_point_estimator.prev_kp_des
        self.initialized += 1
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
            
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose
    
    def add_new_landmarks(self, matches, kp, des, P):
        # TODO: implement
        self.prev_P = P
        
    def estimate(self, img, ransac=True):
        # initialization for first 2 frames (estimate E to init 3d map)
        if self.initialized < 1:
            pose = self.initial_estimation(img)
            
        kp, des = self.tracker.detect(img)
        matches = self.tracker.match(des, np.array([lm.des for lm in self.landmarks]))
        
        if len(matches) < 6: # not enough matches for DLT
            return np.eye(4)
        
        pts3d_homo = np.array([self.landmarks[m.trainIdx].pos for m in matches]).T
        pts2d_euc = np.array([kp[m.queryIdx].pt for m in matches]).T
        pts2d_homo = np.vstack((pts2d_euc, np.ones((1, pts2d_euc.shape[1])))) 
        
        # TODO: estiamte with opencv just to get landmark tracking correct first
        if ransac:
            P = self.dlt_ransac(pts3d_homo, pts2d_homo)
        else:
            P = self.dlt(pts3d_homo, pts2d_homo)        
        pose = self.pose_from_P(P)
        
        self.add_new_landmarks(matches, kp, des, P)
        return pose
        