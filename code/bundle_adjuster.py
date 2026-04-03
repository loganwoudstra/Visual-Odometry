import numpy as np
import cv2

class BundleAdjuster:
    def __init__(self) -> None:
        pass
    
    def adjust(self, pts3d, pts2d, correspondences, cams):
        """
        Args:
            pts3d: np array of shape (4, N)
            pts2d: list of size W, with each elemnt being (3, M_i) np arrays
            correspodneces: list of size W, with each elelmnt being (M_i) list matching each 2d point to a 3d landmark
            cams: np.array of size (W, 4, 4)
        """
        # size check of all variables
        assert len(pts2d) == len(correspondences), f"{len(pts2d)} != {len(correspondences)}"
        assert len(pts2d) == cams.shape[0],  f"{len(pts2d)} != {cams.shape[0]}"
        assert pts3d.shape[0] == 4,  f"{pts3d.shape[0]} != 4"
        assert cams.shape == (len(pts2d), 4, 4), f"{cams.shape} != ({len(pts2d)}, 4, 4)"
        
        W = len(pts2d) # window size
        for i in range(W):
            frame_pts2d = pts2d[i]
            frame_correspondences = correspondences[i]
            assert frame_pts2d.shape[0] == 3, f"{frame_pts2d.shape[0]} != 3"
            assert frame_pts2d.shape[1] == len(frame_correspondences), f"{frame_pts2d.shape[1]} != {len(frame_correspondences)}"
            
        # TODO: do BA
        
        return None, None