from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
from motion_estimation import EssentialMatrixEstimator
    
class EightPointEstimator(EssentialMatrixEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def eight_point(self, pts1, pts2):
        # normalize
        T1 = self.normalize(pts1)
        T2 = self.normalize(pts2)
        pts1_norm = T1 @ pts1
        pts2_norm = T2 @ pts2
        
        N = pts1.shape[1]
        A = np.zeros((N, 9))
        for i in range(N):
            # homogenous to euclidean
            x1, y1 = pts1_norm[:2, i] / pts1_norm[2, i]
            x2, y2 = pts2_norm[:2, i] / pts2_norm[2, i]
            A[i] = (x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1)
            
        _, _, V_t = np.linalg.svd(A)
        F = V_t[-1].reshape(3, 3)
        
        # enforce rank 2 constraint
        Uf, Sf, Vf_t = np.linalg.svd(F)
        Sf[-1] = 0 
        F = Uf @ np.diag(Sf) @ Vf_t
        
        # unnormalize 
        F = T2.T @ F @ T1 
        return F
    
    def sampson_error(self, pts1, pts2, F):
        # sampson error for outlier rejection (distance from epipolar line)
        Fx1 = F @ pts1
        Ftx2 = F.T @ pts2

        numerator = np.sum(pts2 * Fx1, axis=0) ** 2
        denominator = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
        denominator = np.maximum(denominator, 1e-8)

        error = numerator / denominator
        return error
    
    def eight_point_ransac(self, pts1, pts2, tol=1.0, max_iterations=1000, min_inliers=0.75):
        N = pts1.shape[1]
        
        # compute F
        max_inliers = -1
        best_inliers_mask = None
        for i in range(max_iterations):
            sample_ids = np.random.choice(N, 8, replace=False)
            pts1_sample = pts1[:, sample_ids]
            pts2_sample = pts2[:, sample_ids]

            F = self.eight_point(pts1_sample, pts2_sample)
            
            error = self.sampson_error(pts1, pts2, F)
            inliers_mask = error < tol
            n_inliers = inliers_mask.sum()
            
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inliers_mask = inliers_mask
                
            if n_inliers / N >= min_inliers:
                break
        # print(i)
            
        # recompute using all inliers
        inlier_pts1 = pts1[:, best_inliers_mask]
        inlier_pts2 = pts2[:, best_inliers_mask]
        F = self.eight_point(inlier_pts1, inlier_pts2)
        
        return F, best_inliers_mask
    
    def compute_E(self, F):
        E = self.K.T @ F @ self.K
        U, S, V_t = np.linalg.svd(E)
        
        # ensure given F is rank 2
        assert np.isclose(S[-1], 0.0), "F is not rank 2"
        
        # enforce constraint that E is rank 2 AND both singular values are the same
        E = U @ np.diag((1, 1, 0)) @ V_t
        return E
        
    def estimate(self, img, ransac=True, return_pts_des=False):
        self.matches = self.match_features(img)
        if self.matches is None: # first frame
            return np.eye(4)
        pts1, pts2, des = self.tracker.point_correspondences(self.prev_kp, self.prev_des, self.kp, self.des, self.matches)
        
        if ransac:
            F, inlier_mask = self.eight_point_ransac(pts1, pts2)
            pts1 = pts1[:, inlier_mask]
            pts2 = pts2[:, inlier_mask]
        else:
            F = self.eight_point(pts1, pts2)
        E = self.compute_E(F)
        pose = self.pose_from_E(E, pts1, pts2)
        
        if return_pts_des:
            return pose, pts1, pts2, des
        else:
            return pose
    
    
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = EightPointEstimator(dataset.K)
    
    images = iter(dataset.gray)
    
    for i, img in enumerate(images):
        pose = motion_estimator.estimate(img)
        # print(dataset.poses[i + 1])
        # print(pose)
        # print()
        
        # if i % 50 == 0:
        #     print(i)