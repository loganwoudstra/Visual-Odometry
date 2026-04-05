from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
import cv2

class MotionEstimator:
    def __init__(self, K):
        self.K = K
        self.tracker = FeatureTracker()
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
    
    def triangulate_points_batch(self, P1s, P2, pts1, pts2, return_type='homo'):
        x1 = pts1[0] / pts1[2]
        y1 = pts1[1] / pts1[2]
        x2 = pts2[0] / pts2[2]
        y2 = pts2[1] / pts2[2]
        A = np.stack([
            x1[:, None] * P1s[:, 2, :] - P1s[:, 0, :],
            y1[:, None] * P1s[:, 2, :] - P1s[:, 1, :],
            x2[:, None] * P2[2] - P2[0],
            y2[:, None] * P2[2] - P2[1],
        ], axis=1) 

        # batch SVD
        _, _, Vt = np.linalg.svd(A)  # Vt is (N, 4, 4)
        pts3d = Vt[:, -1, :].T # (4, N)
        pts3d = pts3d / pts3d[3] # normalize to have w=1

        if return_type == 'euc':
            return pts3d[:3]
        return pts3d
    
    def reprojection_error(self, pts3d, pts2d, P):
        proj = P @ pts3d
        proj /= proj[2, :] # normalize

        error = np.linalg.norm(proj[:2] - pts2d[:2], axis=0)
        return error
        
    def estimate(self, img):
        raise NotImplementedError("Estimate function not implemented for base class")
    