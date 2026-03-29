import cv2
import matplotlib.pyplot as plt
from dataset import Dataset
import numpy as np

class FeatureTracker:
    def __init__(self):
        # can be orb, sift, etc.
        self.tracker = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def detect(self, img):
        # returns keypoints and descriptions
        return self.tracker.detectAndCompute(img, None)
    
    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def point_correspondences(self, kp_des1, kp_des2, matches):
        kp1, des1 = kp_des1
        kp2, _ = kp_des2
        
        # sue ones for pts so that homogenous w value is set to 1
        pts1 = np.ones((3, len(matches)))
        pts2 = np.ones((3, len(matches)))
        des = np.zeros((len(matches), des1.shape[1]), dtype=des1.dtype)
        for i, match in enumerate(matches):
            x1 = kp1[match.queryIdx].pt
            x2 = kp2[match.trainIdx].pt
            
            des[i] = des1[match.queryIdx]
            pts1[:2, i] = x1
            pts2[:2, i] = x2

        return pts1, pts2, des
            
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    images = iter(dataset.gray)
    img_prev = next(images)
    kp_des_prev = tracker.detect(img_prev)
    
    for i, img in enumerate(images):
        kp_des = tracker.detect(img)
        matches = tracker.match(kp_des_prev[1], kp_des[1])
        pts1, pts2 = tracker.point_correspondences(kp_des_prev[0], kp_des[0], matches)
        
        # feature_match_img = cv2.drawMatchesKnn(img_prev, kp_des_prev[0], img, kp_des[0], matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(feature_match_img)
        # plt.show()
        
        img_prev = img
        kp_des_prev = kp_des
        
        if i % 50 == 0:
            print(i)
    