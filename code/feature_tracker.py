import cv2
import matplotlib.pyplot as plt
from dataset import Dataset
import numpy as np

class FeatureTracker:
    def __init__(self):
        # can be orb, sift, etc.
        self.tracker = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
    def detect(self, img):
        # returns keypoints and descriptions
        return self.tracker.detectAndCompute(img, None)
    
    def match(self, kp_des1, kp_des2):
        kp1, des1 = kp_des1
        kp2, des2 = kp_des2
        
        if des1 is None or des2 is None:
            return []
        
        matches = self.bf.knnMatch(des1, des2, k=2)
 
        # apply ratio test to only get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append([m])
                
        return good_matches
    
    def point_correspondences(self, kp1, kp2, matches):
        # homogenous w value is set to 1
        pts1 = np.ones((3, len(matches)))
        pts2 = np.ones((3, len(matches)))
        for i, match in enumerate(matches):
            m = match[0]
            
            x1 = kp1[m.queryIdx].pt
            x2 = kp2[m.trainIdx].pt
            
            pts1[:2, i] = x1
            pts2[:2, i] = x2

        return pts1, pts2
            
if __name__ == '__main__':
    dataset = Dataset('00')
    tracker = FeatureTracker()
    
    images = iter(dataset.gray)
    img_prev = next(images)
    kp_des_prev = tracker.detect(img_prev)
    
    for i, img in enumerate(images):
        kp_des = tracker.detect(img)
        matches = tracker.match(kp_des_prev, kp_des)
        pts1, pts2 = tracker.point_correspondences(kp_des[0], kp_des_prev[0], matches)
        
        # feature_match_img = cv2.drawMatchesKnn(img_prev, kp_des_prev[0], img, kp_des[0], matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(feature_match_img)
        # plt.show()
        
        img_prev = img
        kp_des_prev = kp_des
        
        if i % 50 == 0:
            print(i)
    