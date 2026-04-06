import numpy as np
from motion_estimation import PnPEstimator

class DLTEstimator(PnPEstimator):
    def __init__(self, K, tracker, window_size=10):
        super().__init__(K, tracker, window_size)
        
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
    
    def _estimate(self, pts3d, pts2d, tol=3.0, max_iterations=1000, min_inliers=0.75):
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
    