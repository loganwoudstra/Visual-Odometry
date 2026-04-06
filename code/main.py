from dataset import Dataset
from motion_estimation import EightPointEstimator, OpenCVPnpEstimator, OpenCVMatrixEstimator, FivePointEstimator, DLTEstimator
from motion_estimation import PnPEstimator, EssentialMatrixEstimator
import errors
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

START_FRAME = 85

def align_trajectories(est_traj, gt_traj, est_poses):
    if len(est_traj) > 2:
        # Umeyama alignment
        s, R_align, t_align = errors.umeyama(est_traj, gt_traj)
        est_aligned_3d = errors.apply_umeyama(est_traj, s, R_align, t_align)  # (N, 3)

        # aligned poses for error computation
        aligned_est_poses = []
        for T in est_poses:
            T_new = T.copy()
            T_new[:3, 3]  = s * R_align @ T[:3, 3] + t_align
            T_new[:3, :3] = R_align @ T[:3, :3]
            aligned_est_poses.append(T_new)

        # 2D for trajectory plot (X-Z)
        est_xz = est_aligned_3d[:, [0, 2]]
        gt_xz  = np.array([[p[0], p[2]] for p in gt_traj])
    else:
        est_xz = np.array([[p[0], p[2]] for p in est_traj])
        gt_xz  = np.array([[p[0], p[2]] for p in gt_traj])
        aligned_est_poses = est_poses
        
    return aligned_est_poses, est_xz, gt_xz

def plot_trajectory(est_xz, gt_xz, point, est_line, gt_line, ax):
    est_line.set_data(est_xz[:, 0], est_xz[:, 1])
    if gt_xz is not None:
        gt_line.set_data(gt_xz[:, 0],   gt_xz[:, 1])
    point.set_data([est_xz[-1, 0]], [est_xz[-1, 1]])
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)
    
def update_frame(img, motion_estimator, mask):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if motion_estimator.matches is not None:
        for i, m in enumerate(motion_estimator.matches):
            if i >= len(mask): # matches that were triangualted this frame
                break
            
            # matched keypoint in current frame
            pt1 = motion_estimator.kp[m.trainIdx].pt
            pt1 = (int(pt1[0]), int(pt1[1]))
        
            if isinstance(motion_estimator, PnPEstimator): # lm projection
                if m.queryIdx not in motion_estimator.landmarks:
                    continue
                lm = motion_estimator.landmarks[m.queryIdx]
                proj = motion_estimator.P @ lm.pos
                proj /= proj[2]
                pt2 = (int(proj[0]), int(proj[1]))
            elif isinstance(motion_estimator, EssentialMatrixEstimator): # prev kp
                pt2 = motion_estimator.prev_kp[m.queryIdx].pt
                pt2 = (int(pt2[0]), int(pt2[1]))

            if not mask[i]:
                cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
                cv2.line(vis, pt2, pt1, (0, 0, 255), 1)
            else:
                cv2.line(vis, pt2, pt1, (0, 255, 0), 1)
                cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
                # cv2.circle(vis, pt2, 3, (255, 0, 0), -1) 
        
    cv2.imshow('cam0', vis)
    cv2.waitKey(1)
    
def plot_errors(aligned_est_poses, gt_poses, est_aligned, gt_traj, ate_rot_line, ate_trans_line, re_line, ax_ate, ax_re, fig_err):
    if len(est_aligned) > 2:
        # ATE — scale=1.0 since alignment already applied
        ate_rot, ate_trans, rot_errors, trans_errors = errors.ate(aligned_est_poses, gt_poses, scale=1.0)
        frames = list(range(len(est_aligned)))
        ate_rot_line.set_data(frames, rot_errors)
        ate_trans_line.set_data(frames, trans_errors)
        ax_ate.set_title(f"ATE — rot: {ate_rot:.2f}° | trans: {ate_trans:.2f}m")
        ax_ate.relim(); ax_ate.autoscale_view()

        # RE
        distance_thresholds = [1, 2, 5, 10, 20]
        re_stats, _ = errors.re(aligned_est_poses, gt_poses, scale=1.0, distance_thresholds=distance_thresholds)
        if re_stats:
            dists   = [d for d in distance_thresholds if d in re_stats]
            re_meds = [re_stats[d]['pos_median'] for d in dists]
            re_line.set_data(dists, re_meds)
            ax_re.relim(); ax_re.autoscale_view()

        fig_err.canvas.draw()
        fig_err.canvas.flush_events()
    
def main():
    parser = argparse.ArgumentParser(description='Visual odometry.')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence from KITTTI dataset')
    parser.add_argument('--method', type=str, default='eightpoint', help='Pose estimation method')
    parser.add_argument('--tracker', type=str, default='orb', help='Tracker method')
    parser.add_argument('--slow', action='store_true', help='Stop between frames')
    parser.add_argument('--errors', action='store_true', help='Plot errors and comapre to ground truth')
    args = parser.parse_args()
    
    sequence = args.sequence
    method = args.method
    tracker = args.tracker
    slow = args.slow
    should_plot_errors = args.errors
    
    dataset = Dataset(sequence)
    K = dataset.K

    match method:
        case 'eightpoint':
            motion_estimator = EightPointEstimator(K, tracker)
        case 'fivepoint':
            motion_estimator = FivePointEstimator(K, tracker)
        case 'dlt':
            motion_estimator = DLTEstimator(K, tracker)
        case 'pnp':
            motion_estimator = OpenCVPnpEstimator(K, tracker)
        case 'opencv_matrix':
            motion_estimator = OpenCVMatrixEstimator(K, tracker)
        case _:
            raise Exception(f'Invalid method: {method}')
        
    images = iter(dataset.gray)
    for _ in range(START_FRAME):
        next(images)
    
    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(4, 3))
    est_line, = ax.plot([], [], 'g-')
    point, = ax.plot([], [], 'ro') # current pos
    gt_line, = ax.plot([], [], 'b-', label='ground truth')
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Trajectory")
    ax.axis("equal")
    est_line.set_label('estimated')
    ax.legend()
    
    gt_poses = dataset.poses
    
    # error figure
    if should_plot_errors:
        fig_err, (ax_ate, ax_re) = plt.subplots(1, 2, figsize=(8, 4))
        
        ate_rot_line, = ax_ate.plot([], [], 'r-',  label='ATE rot (deg)')
        ate_trans_line, = ax_ate.plot([], [], 'b-',  label='ATE trans (m)')
        ax_ate.set_xlabel("Frame"); ax_ate.set_ylabel("Error")
        ax_ate.set_title("Absolute Trajectory Error"); ax_ate.legend()

        re_line, = ax_re.plot([], [], 'g-', label='RE pos median (m)')
        ax_re.set_xlabel("Distance threshold (m)")
        ax_re.set_ylabel("Relative Error (m)")
        ax_re.set_title("Relative Error by Distance")
        ax_re.legend()
    
    # if slow:
    input('...')
    for i, img in enumerate(images):
        if i > 0 and i < 5:
            continue
        mask = motion_estimator.step(img)
        
        est_poses = motion_estimator.trajectory
        est_traj = [p[:3, 3] for p in est_poses]
        
        if should_plot_errors:
            n_frames = len(est_poses)
            gt_poses_curr = [gt_poses[START_FRAME + j] for j in range(n_frames)]
            gt_traj = [p[:3, 3] for p in gt_poses_curr]
            aligned_est_poses, est_xz, gt_xz = align_trajectories(est_traj, gt_traj, est_poses)
        else:
            est_xz = np.array([[p[0], p[2]] for p in est_traj])
            gt_xz = None

        plot_trajectory(est_xz, gt_xz, point, est_line, gt_line, ax)
        if should_plot_errors:
            plot_errors(aligned_est_poses, gt_poses_curr, est_xz, gt_xz, ate_rot_line, ate_trans_line, re_line, ax_ate, ax_re, fig_err)
        update_frame(img, motion_estimator, mask)
        
        if slow:
            input('...')
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()