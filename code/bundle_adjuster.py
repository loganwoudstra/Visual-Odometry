import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

class BundleAdjuster:
    def __init__(self, K) -> None:
        self.K = K
        self.W = 0
        self.N = 0
        self.correspondences = []
        self.pts2d_euc = []
        self.cam0 = np.zeros(6)
    
    def residuals(self, x):
        free_cam_params = x[:( self.W - 1)*6].reshape(self.W - 1, 6)
        cam_params = np.vstack([self.cam0[None, :], free_cam_params])
        pts3d = x[(self.W-1)*6:].reshape(self.N, 3)
        
        rvecs = cam_params[:, :3]  # (W,3)
        tvecs = cam_params[:, 3:]  # (W,3)

        # convert all rvecs to rotation matrices at once
        rots = Rotation.from_rotvec(rvecs)
        R_all = rots.as_matrix()

        residuals = []
        for cam_idx in range(self.W):
            pts_idx = self.correspondences[cam_idx]
            pts3d_i = pts3d[pts_idx]  # (Mi, 3)
            pts_cam = (R_all[cam_idx] @ pts3d_i.T).T + tvecs[cam_idx]  # (Mi, 3)
            
            proj = (self.K @ pts_cam.T).T
            proj2d = proj[:, :2] / proj[:, 2:3]

            residuals.append((proj2d - self.pts2d_euc[cam_idx]).reshape(-1))
        return np.concatenate(residuals)
    
    def unpack_result(self, result):
        x_opt = result.x
        free_cam_params = x_opt[:(self.W-1)*6].reshape(self.W - 1, 6)
        cam_params_opt = np.vstack([self.cam0[None, :], free_cam_params])
        pts3d_opt_euc = x_opt[(self.W-1)*6:].reshape(self.N, 3)

        # convert 6DOF (R, t) to 4x4 camera pose
        new_cams = np.zeros((self.W, 4, 4))
        for i in range(self.W):
            rvec = cam_params_opt[i, :3]
            tvec = cam_params_opt[i, 3:]

            R, _ = cv2.Rodrigues(rvec)
            R_wc = R.T
            t_wc = -R.T @ tvec

            pose = np.eye(4)
            pose[:3, :3] = R_wc
            pose[:3, 3] = t_wc
            new_cams[i] = pose

        # euc to homo
        pts3d_opt_homo = np.vstack([pts3d_opt_euc.T, np.ones((1, self.N))])  # (4, N)

        return pts3d_opt_homo, new_cams
    
    def adjust(self, pts3d_homo, pts2d_homo, correspondences, cams):
        """
        Args:
            pts3d: np array of shape (4, N)
            pts2d: list of size W, with each elemnt being (3, M_i) np arrays
            correspodneces: list of size W, with each elelmnt being (M_i) list matching each 2d point to a 3d landmark
            cams: np.array of size (W, 4, 4)
        """
        # size check of all variables
        self.W = len(pts2d_homo) # window size
        self.N = pts3d_homo.shape[1] # number of landmarks
        
        assert len(correspondences) == self.W, f"{len(correspondences)} !=  {self.W}"
        assert cams.shape[0] == self.W,  f"{cams.shape[0]} != {self.W}"
        assert pts3d_homo.shape[0] == 4,  f"{pts3d_homo.shape[0]} != 4"
        assert cams.shape == (self.W, 4, 4), f"{cams.shape} != ({self.W}, 4, 4)"
        
        for i in range(self.W):
            frame_pts2d = pts2d_homo[i]
            frame_correspondences = correspondences[i]
            assert frame_pts2d.shape[0] == 3, f"{frame_pts2d.shape[0]} != 3"
            assert frame_pts2d.shape[1] == len(frame_correspondences), f"{frame_pts2d.shape[1]} != {len(frame_correspondences)}"
            
        # convert 4x4 camera pose to 6 DOF (R, t)
        cam_params = np.zeros((self.W, 6))
        for i in range(self.W):
            pose = cams[i]
            R_wc = pose[:3, :3]
            t_wc = pose[:3, 3]

            R = R_wc.T
            t = -R @ t_wc

            rvec, _ = cv2.Rodrigues(R)
            cam = np.hstack([rvec.flatten(), t])
            cam_params[i] = cam
        
        # NOTE: need to pack all params together and use class vars bc loss function (residuals) only takes 1 vector input of optimizable params
        pts3d_euc = (pts3d_homo[:3] / pts3d_homo[3]).T  # (N, 3)
        
        self.pts2d_euc = [] # (W, Mi, 2)
        for i in range(self.W):
            pts2d_euc = (pts2d_homo[i][:2] / pts2d_homo[i][2]).T  # (Mi, 2)
            self.pts2d_euc.append(pts2d_euc)
            
        self.correspondences = correspondences
        self.cam0 = cam_params[0]
        
        total_obs = sum(len(c) for c in correspondences)
        sparse_J = lil_matrix((2*total_obs, 6*(self.W-1) + 3*self.N))
        row_idx = 0
        for cam_idx, pts_idx in enumerate(correspondences):
            for pt_idx in pts_idx:
                if cam_idx > 0:  # cam0 columns don't exist in J
                    sparse_J[row_idx:row_idx+2, 6*(cam_idx-1) : 6*(cam_idx-1)+6] = 1
                sparse_J[row_idx:row_idx+2, 6*(self.W-1) + 3*pt_idx : 6*(self.W-1) + 3*pt_idx+3] = 1
                row_idx += 2
        
        x0 = np.hstack([
            cam_params[1:].flatten(),
            pts3d_euc.flatten()
        ])
        
        # optimization
        result = least_squares(
            self.residuals,
            x0,
            verbose=2,
            method='trf',
            loss='huber',
            f_scale=3.0,       
            jac_sparsity=sparse_J.tocsr(), 
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=1000
        )
        
        pts3d_homo, cams = self.unpack_result(result)
        return pts3d_homo, cams