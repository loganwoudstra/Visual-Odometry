import cv2
from feature_tracking import FeatureTracker

class DescriptionMatcher(FeatureTracker):
    def __init__(self):
        super().__init__()
        self.prev_des = None
        self.des = None
        
        # can be orb, sift, etc.
        self.tracker = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # self.tracker = cv2.SIFT_create()
        
    def detect(self, img):
        # returns keypoints and descriptions
        return self.tracker.detectAndCompute(img, None)
    
    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    # def match(self, des1, des2, ratio=0.75):
    #     raw = self.bf.knnMatch(des1, des2, k=2)
    #     good = []
    #     for pair in raw:
    #         if len(pair) == 2:
    #             m, n = pair
    #             if m.distance < ratio * n.distance:
    #                 good.append(m)
    #     return good
    
    def get_matches(self, img):
        self.prev_kp = self.kp
        self.prev_des = self.des
        self.kp, self.des = self.detect(img)
        
        if self.prev_kp is None: # first frame
            matches = []
        else:
            matches = self.match(self.prev_des, self.des)
            
        self.prev_matches = matches
        return self.kp, matches