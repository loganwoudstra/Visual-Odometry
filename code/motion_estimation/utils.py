import numpy as np

def unproject_many(pts2d, K):
        x = (pts2d[0] - K[0, 2]) / K[0, 0]
        y = (pts2d[1] - K[1, 2]) / K[1, 1]
        rays = np.vstack([x, y, np.ones(pts2d.shape[1])])
        return rays / np.linalg.norm(rays, axis=0)
    
def bearing_angles(pts1_homo, pts2_homo, P1s, P2, K):
    N = pts1_homo.shape[1]
    assert pts1_homo.shape == (3, N)
    assert pts2_homo.shape == (3, N)
    assert P1s.shape == (N, 3, 4)
    assert P2.shape == (3, 4)
    
    # unproject to bearing vectors in each camera's frame
    f1 = unproject_many(pts1_homo[:2], K)
    f2 = unproject_many(pts2_homo[:2], K)

    # rotate both rays into world frame
    R1s = P1s[:, :3, :3] 
    R2  = P2[:3, :3]
    f1_world = np.einsum('nij,jn->in', R1s, f1)
    f2_world = R2.T @ f2 

    cos_angles = np.clip(np.einsum('in,in->n', f1_world, f2_world), -1.0, 1.0)
    return np.arccos(cos_angles)

def camera_center(P, K):
    M = np.linalg.inv(K) @ P  # 3x4 [R|t]
    R = M[:, :3]
    t = M[:, 3]
    return -R.T @ t  # 3D world position of camera