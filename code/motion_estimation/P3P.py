from dataset import Dataset
from feature_tracker import FeatureTracker
import numpy as np
from motion_estimation import MotionEstimator, EightPointEstimator


class P3PEstimator(MotionEstimator):
    def __init__(self, K):
        super().__init__(K)
        self.eight_point_estimator = EightPointEstimator(self.K)
        
    def estimate(self, pts1, pts2):
        assert pts1.shape == pts2.shape, "pts1 and pts2 of different shapes"
    
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = P3PEstimator(dataset.K)
    
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