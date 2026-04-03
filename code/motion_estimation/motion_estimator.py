from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
import cv2

class MotionEstimator:
    def __init__(self, K):
        self.K = K
        self.tracker = FeatureTracker()
        self.returns_global = False
        self.trajectory = []
        self.kp = None
        self.des = None
        self.prev_kp =None
        self.prev_des = None
        self.matches = None
    
    def normalize(self, pts):
        """normalizes 2d homogenous or 3d homogenous"""
        D = pts.shape[0] - 1  # dimension

        # homo to euclidean
        X = pts[:D, :] / pts[D, :]

        # 0 center
        mean = X.mean(axis=1)
        X_centered = X - mean[:, None]

        # root(D) avg dist
        dists = np.linalg.norm(X_centered, axis=0)
        avg_dist = dists.mean()
        scale = np.sqrt(D) / avg_dist

        # Build matrix
        T = np.eye(D + 1)
        T[:D, :D] *= scale
        T[:D, D] = -scale * mean

        return T
        
    def triangulate_points(self, P1, P2, pts1, pts2, return_type='homo'):
        N = pts1.shape[1]
        pts_3d = np.zeros((4, N))
        for i in range(N):
            x1, y1 = pts1[:2, i] / pts1[2, i]
            x2, y2 = pts2[:2, i] / pts2[2, i]
            A = np.array([
                x1*P1[2] - P1[0],
                y1*P1[2] - P1[1],
                x2*P2[2] - P2[0],
                y2*P2[2] - P2[1],
            ])
            _, _, V_t = np.linalg.svd(A)
            X = V_t[-1]
            X = X / X[3] # normalize to have w=1
            pts_3d[:, i] = X
            
        if return_type == 'euc':
            pts_3d = pts_3d[:3]
        return pts_3d
        
    def estimate(self, img):
        raise NotImplementedError("Estimate function not implemented for base class")
    
    def step(self, img):
        pose = self.estimate(img)
        if self.returns_global:
            global_pose = pose
        else:
            last_global_pos = self.trajectory[-1] if self.trajectory else np.eye(4) 
            global_pose = last_global_pos @ pose
        self.trajectory.append(global_pose)
    
class EssentialMatrixEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        self.returns_global = False
        
    def match_features(self, img):
        self.prev_kp = self.kp
        self.prev_des = self.des
        self.kp, self.des = self.tracker.detect(img)
        if self.prev_kp is None or self.prev_des is None: # first frame
            return None
        matches = self.tracker.match(self.prev_des, self.des)
        return matches
        
    def pose_from_E(self, E, pts1, pts2):
        U, S, V_t = np.linalg.svd(E)
        assert (np.abs(S[0] - S[1]) < 1e-6) and (np.abs(S[2]) < 1e-6), f"Singular values not of the form (a, a, 0): {S}"
        
        if np.linalg.det(U @ V_t) < 0: # bc rotation matrices must have det = +1
            V_t = -V_t

        W = np.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ])
        
        # get 4 candidtae solutions
        R1 = U @ W @ V_t
        R2 = U @ W.T @ V_t
        t = U[:, -1]
        t = t / np.linalg.norm(t)
        candidates = [
            (R1, t),
            (R1, -t),
            (R2, t),
            (R2, -t)
        ]
        
        P1 = self.K @ np.hstack((np.eye(3), np.zeros(3).reshape(3, 1)))
        R_best = np.eye(3)
        t_best = np.zeros(3)
        max_inliers = -1
        for R, t in candidates:
            P2 = self.K @ np.hstack([R, t.reshape(3,1)])
            pts_euc = self.triangulate_points(P1, P2, pts1, pts2, return_type='euc')
            inlier_count = 0
            for pt in pts_euc.T:
                cam1_depth = pt[2]
                cam2_pt = R @ pt[:3] + t
                cam2_depth = cam2_pt[2]
                if cam1_depth > 0 and cam2_depth > 0:
                    inlier_count += 1
            if inlier_count > max_inliers:
                R_best = R
                t_best = t
                max_inliers = inlier_count
        
        # covnert R, t into homogenous matrix
        pose = np.eye(4)
        pose[:3, :3] = R_best.T # R_wc = R_cw.T
        pose[:3, 3] = -R_best.T @ t_best 
        
        return pose
    
class OpenCVEstimator(EssentialMatrixEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def estimate(self, img):
        self.matches = self.match_features(img)
        if self.matches is None: # first frame
            return np.eye(4)
        pts1, pts2 = self.tracker.point_correspondences(self.prev_kp, self.prev_des, self.kp, self.des, self.matches)
        
        E, mask = cv2.findEssentialMat(pts1[:2].T, pts2[:2].T, self.K, cv2.RANSAC)
        mask = mask.ravel().astype(bool)
        pose = self.pose_from_E(E, pts1[:, mask], pts2[:, mask])
        
        return pose
    
    
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = OpenCVEstimator(dataset.K)
    
    images = iter(dataset.gray)
    img_prev = next(images)
    kp_des_prev = tracker.detect(img_prev)
    
    for i, img in enumerate(images):
        pose = motion_estimator.estimate(img)
        # print(dataset.poses[i + 1])
        # print(pose)
        # print()
        
        # if i % 50 == 0:
        #     print(i)