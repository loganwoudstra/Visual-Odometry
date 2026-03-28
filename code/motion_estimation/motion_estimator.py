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
                max_inliers = inlier_count
        
        # covnert R, t into homogenous matrix
        pose = np.eye(4)
        pose[:3, :3] = R_best
        pose[:3, 3] = t_best
        
        return pose
        
    def estimate(self, pts1, pts2):
        raise NotImplementedError("Estimate function not implemented for base class")
    
class OpenCVEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def estimate(self, pts1, pts2):
        assert pts1.shape == pts2.shape, "pts1 and pts2 of different shapes"
        
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
        kp_des = tracker.detect(img)
        matches = tracker.match(kp_des_prev, kp_des)
        pts1, pts2 = tracker.point_correspondences(kp_des[0], kp_des_prev[0], matches)
        
        pose = motion_estimator.estimate(pts1, pts2)
        # print(dataset.poses[i + 1])
        # print(pose)
        # print()

        img_prev = img
        kp_des_prev = kp_des
        
        # if i % 50 == 0:
        #     print(i)