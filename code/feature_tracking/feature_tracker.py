class FeatureTracker:
    def __init__(self):
        self.prev_kp = None
        self.prev_matches = None
        self.kp = None
        
    def get_matches(self, img):
        raise NotImplementedError