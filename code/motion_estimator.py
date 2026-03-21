from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
import cv2

class MotionEstimator:
    def __init__(self, K):
        self.K = K
    
    def normalize(self, pts):
        # go from homogenous to euclidean
        x = pts[0, :] / pts[2, :]
        y = pts[1, :] / pts[2, :]
        
        # (0, 0) mean
        mean_x = x.mean()
        mean_y = y.mean()
        
        # root 2 avg dist
        avg_dist = np.sqrt((x - mean_x)**2 + (y - mean_y)**2).mean()
        scale = np.sqrt(2) / avg_dist
        
        T = np.array([
            [scale, 0., -scale * mean_x],
            [0., scale, -scale * mean_y],
            [0., 0., 1.]
        ])
        
        return T
    
    def eight_point(self, pts1, pts2):
        N = pts1.shape[0]
        A = np.zeros((N, 9))
        for i in range(N):
            # homogenous to euclidean
            x1, y1 = pts1[:2, i] / pts1[2, i]
            x2, y2 = pts2[:2, i] / pts2[2, i]
            A[i] = (x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1)
            
        _, _, V_t = np.linalg.svd(A)
        F = V_t[-1].reshape(3, 3)
        
        Uf, Sf, Vf_t = np.linalg.svd(F)
        Sf[-1] = 0 # enforce rank 2 constraint
        F_rank2 = Uf @ np.diag(Sf) @ Vf_t
        return F_rank2
    
    def normalized_eight_point(self, pts1, pts2):
        # normalize
        T1 = self.normalize(pts1)
        T2 = self.normalize(pts2)
        normalized_pts1 = T1 @ pts1
        normalized_pts2 = T2 @ pts2
        
        # compute F
        F = self.eight_point(normalized_pts1, normalized_pts2)
        
        # unnormalize
        F = T2.T @ F @ T1 
        return F
    
    def eight_point_ransac(self, pts1, pts2, tol=1.5, max_iterations=1000, min_inliers=0.90):
        # normalize
        T1 = self.normalize(pts1)
        T2 = self.normalize(pts2)
        normalized_pts1 = T1 @ pts1
        normalized_pts2 = T2 @ pts2
        N = pts1.shape[1]
        
        # compute F
        max_inliers = -1
        best_inliers_mask = None
        for i in range(max_iterations):
            sample_ids = np.random.choice(N, 8, replace=False)
            pts1_sample = normalized_pts1[:, sample_ids]
            pts2_sample = normalized_pts2[:, sample_ids]
            F = self.eight_point(pts1_sample, pts2_sample)
            
            # epipoalr constraint x'Fx = 0
            constraint_vals = np.sum(normalized_pts2 * (F @ normalized_pts1), axis=0)
            inliers_mask  = np.abs(constraint_vals) < tol
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
        
        # unnormalize
        F = T2.T @ F @ T1 
        return F, best_inliers_mask
    
    def compute_E(self, F):
        E = self.K.T @ F @ self.K
        U, S, V_t = np.linalg.svd(E)
        
        # ensure given F is rank 2
        assert np.isclose(S[-1], 0.0), "F is not rank 2"
        
        # enforce constraint that E is rank 2 AND both singular values are the same
        E = U @ np.diag((1, 1, 0)) @ V_t
        return E
    
    def triangulate_points(self, P1, P2, pts1, pts2):
        N = pts1.shape[1]
        pts_3d = np.zeros((3, N))
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
            X = X / X[3] # homo to euclidean
            pts_3d[:, i] = X[:3]
        return pts_3d
    
    def pose_from_E(self, E, pts1, pts2):
        U, S, V_t = np.linalg.svd(E)
        assert np.allclose(S, [1., 1., 0.,]), "Essential matrix does not have singular values (1, 1, 0)"
        
        W = np.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ])
        
        # get 4 candidtae solutions
        R1 = U @ W @ V_t
        R2 = U @ W.T @ V_t
        if np.linalg.det(R1) < 0: # makes sure rotation matrices have pos determinants
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2
        t = U[:, -1]
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
            pts_3d = self.triangulate_points(P1, P2, pts1, pts2)
            inlier_count = 0
            for pt in pts_3d.T:
                cam1_depth = pt[2]
                cam2_pt = R @ pt[:3] + t
                cam2_depth = cam2_pt[2]
                if cam1_depth > 0 and cam2_depth > 0:
                    inlier_count += 1
            if inlier_count > max_inliers:
                R_best = R
                t_best = t
        
        # covnert R, t into homogenous matrix
        pose = np.eye(4)
        pose[:3, :3] = R_best
        pose[:3, 3] = t_best
        
        return pose
        
    def estimate(self, pts1, pts2, ransac=False):
        assert pts1.shape == pts2.shape, "pts1 and pts2 of different shapes"
        if ransac:
            F, inlier_mask = self.eight_point_ransac(pts1, pts2)
            pts1 = pts1[:, inlier_mask]
            pts2 = pts2[:, inlier_mask]
        else:
            F = self.normalized_eight_point(pts1, pts2)
        E = self.compute_E(F)
        pose = self.pose_from_E(E, pts1, pts2)
        
        return pose
    
    
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = MotionEstimator(dataset.K)
    
    images = iter(dataset.gray)
    img_prev = next(images)
    kp_des_prev = tracker.detect(img_prev)
    
    for i, img in enumerate(images):
        kp_des = tracker.detect(img)
        matches = tracker.match(kp_des_prev, kp_des)
        pts1, pts2 = tracker.point_correspondences(kp_des[0], kp_des_prev[0], matches)
        
        pose = motion_estimator.estimate(pts1, pts2, ransac=True)
        # print(dataset.poses[i + 1])
        # print(pose)
        # print()

        img_prev = img
        kp_des_prev = kp_des
        
        # if i % 50 == 0:
        #     print(i)