import numpy as np

def umeyama(est_traj, gt_traj):
    est = np.array([[p[0], p[1], p[2]] for p in est_traj])  # (N, 3)
    gt  = np.array([[p[0], p[1], p[2]] for p in gt_traj])   # (N, 3)
    N = len(est)

    mu_est = est.mean(axis=0)
    mu_gt  = gt.mean(axis=0)

    est_centered = est - mu_est
    gt_centered  = gt  - mu_gt

    var_est = np.mean(np.sum(est_centered**2, axis=1))

    Sigma = (gt_centered.T @ est_centered) / N

    U, D, Vt = np.linalg.svd(Sigma)

    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1

    R = U @ W @ Vt
    s = np.trace(np.diag(D) @ W) / var_est
    t = mu_gt - s * R @ mu_est

    return s, R, t

def apply_umeyama(est_traj, s, R, t):
    """Apply alignment to trajectory. Returns (N, 3)."""
    est = np.array([[p[0], p[1], p[2]] for p in est_traj])
    return (s * (R @ est.T)).T + t

def ate(est_poses, gt_poses, scale):
    N = len(est_poses)
    rot_errors   = np.zeros(N)
    trans_errors = np.zeros(N)

    for i in range(N):
        R_gt = gt_poses[i][:3, :3]
        p_gt = gt_poses[i][:3, 3]

        R_est = est_poses[i][:3, :3]
        p_est = est_poses[i][:3, 3] * scale

        # rotation error
        dR = R_gt @ R_est.T
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        dp = np.linalg.norm(p_gt - dR @ p_est)
        
        rot_errors[i] = np.degrees(angle)
        trans_errors[i] = dp

    ate_rot = np.sqrt(np.mean(rot_errors**2))
    ate_trans = np.sqrt(np.mean(trans_errors**2))
    return ate_rot, ate_trans, rot_errors, trans_errors

def re(est_poses, gt_poses, scale, distance_thresholds=[1, 2, 5, 10, 20]):
    N = len(est_poses)

    # precompute cumulative distances along gt trajectory
    gt_positions = np.array([p[:3, 3] for p in gt_poses])
    cum_dists = np.zeros(N)
    for i in range(1, N):
        cum_dists[i] = cum_dists[i-1] + np.linalg.norm(gt_positions[i] - gt_positions[i-1])

    results = {d: {'rot': [], 'pos': []} for d in distance_thresholds}

    for s in range(N):
        for d in distance_thresholds:
            # find end index e such that gt distance from s to e is ~d
            target_dist = cum_dists[s] + d
            if target_dist > cum_dists[-1]:
                continue
            e = np.searchsorted(cum_dists, target_dist)
            if e >= N:
                continue

            # alignment: T_align = gt_s @ inv(est_s)
            est_s = est_poses[s].copy()
            est_s[:3, 3] *= scale
            est_e = est_poses[e].copy()
            est_e[:3, 3] *= scale

            gt_s = gt_poses[s]
            gt_e = gt_poses[e]

            T_align = gt_s @ np.linalg.inv(est_s)

            # aligned estimated end state
            est_e_aligned = T_align @ est_e

            R_gt_e    = gt_e[:3, :3]
            p_gt_e    = gt_e[:3, 3]
            R_est_e   = est_e_aligned[:3, :3]
            p_est_e   = est_e_aligned[:3, 3]

            # error at end
            dR  = R_gt_e @ R_est_e.T
            angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            dp  = np.linalg.norm(p_gt_e - dR @ p_est_e)

            results[d]['rot'].append(angle)
            results[d]['pos'].append(dp)

    # compute statistics per distance threshold
    stats = {}
    for d in distance_thresholds:
        if results[d]['rot']:
            stats[d] = {
                'rot_median':  np.median(results[d]['rot']),
                'rot_mean':    np.mean(results[d]['rot']),
                'pos_median':  np.median(results[d]['pos']),
                'pos_mean':    np.mean(results[d]['pos']),
            }
    return stats, results