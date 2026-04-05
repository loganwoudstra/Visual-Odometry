import numpy as np
import cv2
from motion_estimation import MotionEstimator
from feature_tracker import FeatureTracker
from dataset import Dataset

class EssentialMatrixEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def match_features(self, img):
        self.prev_kp = self.kp
        self.prev_des = self.des
        self.kp, self.des = self.tracker.detect(img)
        if self.prev_kp is None or self.prev_des is None: # first frame
            return None
        matches = self.tracker.match(self.prev_des, self.des)
        return matches
    
    def sampson_error(self, pts1, pts2, F):
        # sampson error for outlier rejection (distance from epipolar line)
        Fx1 = F @ pts1
        Ftx2 = F.T @ pts2

        numerator = np.sum(pts2 * Fx1, axis=0) ** 2
        denominator = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
        denominator = np.maximum(denominator, 1e-8)

        error = numerator / denominator
        return error
    
    def E_from_F(self, F):
        E = self.K.T @ F @ self.K
        U, S, V_t = np.linalg.svd(E)
        
        # ensure given F is rank 2
        assert np.isclose(S[-1], 0.0), "F is not rank 2"
        
        # enforce constraint that E is rank 2 AND both singular values are the same
        E = U @ np.diag((1, 1, 0)) @ V_t
        return E
        
    def pose_from_E(self, E, pts1, pts2):
        U, S, V_t = np.linalg.svd(E)
        assert (np.abs(S[0] - S[1]) < 1e-3) and (np.abs(S[2]) < 1e-3), f"Singular values not of the form (a, a, 0): {S}"
        
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
        best_mask = None
        for R, t in candidates:
            P2 = self.K @ np.hstack([R, t.reshape(3,1)])
            pts_euc = self.triangulate_points(P1, P2, pts1, pts2, return_type='euc')
            inlier_count = 0
            
            cam1_depths = pts_euc[2]
            cam2_depths = (R @ pts_euc + t.reshape(3, 1))[2]
            cheirality_mask = (cam1_depths > 0) & (cam2_depths > 0)
            inlier_count = cheirality_mask.sum()
                    
            if inlier_count > max_inliers:
                R_best = R
                t_best = t
                max_inliers = inlier_count
                best_mask = cheirality_mask
        
        # covnert R, t into homogenous matrix
        pose = np.eye(4)
        pose[:3, :3] = R_best.T # R_wc = R_cw.T
        pose[:3, 3] = -R_best.T @ t_best 
        
        return pose, best_mask
    
    def estimate(self, img, return_pts=False):
        self.matches = self.match_features(img)
        if self.matches is None: # first frame
            return np.eye(4), []
        pts1, pts2 = self.tracker.point_correspondences(self.prev_kp, self.kp, self.matches)
        pose, mask = self._estimate(pts1, pts2)
        
        pts1 = pts1[:, mask]
        pts2 = pts2[:, mask]
        self.matches = [m for m, keep in zip(self.matches, mask) if keep]
        
        if return_pts:
            return pose, mask, pts1, pts2
        else:
            return pose, mask
    
    def _estimate(self, pts1, pts2):
        raise NotImplementedError
    
    def step(self, img):
        rel_pose, mask = self.estimate(img)
        last_global_pose = self.trajectory[-1] if self.trajectory else np.eye(4)
        global_pose = last_global_pose @ rel_pose
        self.trajectory.append(global_pose)
        return mask
    
class OpenCVMatrixEstimator(EssentialMatrixEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def _estimate(self, pts1, pts2):
        E, inlier_mask = cv2.findEssentialMat(pts1[:2].T, pts2[:2].T, self.K, cv2.RANSAC)
        inlier_mask = inlier_mask.ravel().astype(bool)
        pose, cheirality_mask  = self.pose_from_E(E, pts1[:, inlier_mask], pts2[:, inlier_mask])
        
        full_mask = np.zeros(pts1.shape[1], dtype=bool)
        full_mask[inlier_mask] = cheirality_mask
        return pose, full_mask
    
    
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = OpenCVMatrixEstimator(dataset.K)
    
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