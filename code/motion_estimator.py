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
    
    def eight_points(self, pts1, pts2):
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
    
    def normalized_eight_points(self, pts1, pts2):
        assert pts1.shape == pts2.shape, "pts1 and pts2 of different shapes" 
        
        # normalize
        T1 = self.normalize(pts1)
        T2 = self.normalize(pts2)
        pts1 = T1 @ pts1
        pts2 = T2 @ pts2
        
        # compute F
        F = self.eight_points(pts1, pts2)
        
        # unnormalize
        F = T2.T @ F @ T1 
        return F
    
    def compute_E(self, F):
        E = self.K.T @ F @ self.K
        U, S, V_t = np.linalg.svd(E)
        
        # ensure given F is rank 2
        assert np.isclose(S[-1], 0.0), "F is not rank 2"
        
        # enforce constraint that E is rank 2 AND both singular values are the same
        E = U @ np.diag((1, 1, 0)) @ V_t
        return E
    
    def pose_from_E(self, E):
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
        
        P1 = self.K @ np.hstack((np.eye(3), np.zeros(3)))
        for R, t in candidates:
            P2 = self.K @ np.hstack([R, t.reshape(3,1)])
            # TODO: triangulate points
        
        
    
    def estimate(self, pts1, pts2, ransac=False):
        F = self.normalized_eight_points(pts1, pts2)
        E = self.compute_E(F)
        R, t = self.pose_from_E(E)
        
        return R, t
    
    
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
        
        R, t = motion_estimator.estimate(pts1, pts2)

        img_prev = img
        kp_des_prev = kp_des
        
        if i % 50 == 0:
            print(i)