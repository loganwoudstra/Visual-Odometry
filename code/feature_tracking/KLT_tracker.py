import cv2
import numpy as np

from feature_tracking import FeatureTracker

class KLTTracker(FeatureTracker):
    def __init__(self, max_corners=500, quality_level=0.1, min_distance=12):
        super().__init__()
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_img = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.05)
        )

    def detect(self, img, mask=None):
        corners = cv2.goodFeaturesToTrack(
            img, self.max_corners, self.quality_level, self.min_distance, mask=mask
        )
        if corners is None:
            return []
        return [cv2.KeyPoint(p[0][0], p[0][1], 1) for p in corners]

    def get_matches(self, img):
        self.prev_kp = self.kp
        
        if self.prev_img is None or self.kp is None or len(self.kp) == 0:
            self.kp = self.detect(img)
            self.prev_img = img
            return self.kp, []  # first frame, no matches

        prev_pts = np.array([kp.pt for kp in self.prev_kp], dtype=np.float32).reshape(-1, 1, 2) # opencv wants (N, 1, 2)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, prev_pts, None, **self.lk_params
        )
        status = status.flatten().astype(bool)
        self.kp = [cv2.KeyPoint(curr_pts[i][0][0], curr_pts[i][0][1], 1) for i in range(len(self.prev_kp)) if status[i]]

        # always detect new points, masking out existing ones
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        for kp in self.kp:
            cv2.circle(mask, (int(kp.pt[0]), int(kp.pt[1])), self.min_distance, 0, -1)
        new_kp = self.detect(img, mask=mask)
        self.kp += new_kp

        self.prev_img = img
        tracked_prev = [i for i in range(len(self.prev_kp)) if status[i]]
        matches = [cv2.DMatch(tracked_prev[j], j, 0.0) for j in range(len(tracked_prev))]
        return self.kp, matches