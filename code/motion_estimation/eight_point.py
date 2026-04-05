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
        
    def _estimate(self, pts1, pts2):
        F, mask = self.eight_point_ransac(pts1, pts2)
        E = self.E_from_F(F)
        pose = self.pose_from_E(E, pts1[:, mask], pts2[:, mask])
        return pose, mask
    
    
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