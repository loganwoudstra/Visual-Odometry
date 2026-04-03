from dataset import Dataset
from motion_estimation import EightPointEstimator, DLTEstimator, OpenCVEstimator
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(sequence, method):
    dataset = Dataset(sequence)
    if method == 'eightpoint':
        motion_estimator = EightPointEstimator(dataset.K)
    elif method == 'dlt':
        motion_estimator = DLTEstimator(dataset.K)
    elif method == 'opencv':
        motion_estimator = OpenCVEstimator(dataset.K)
    else:
        raise Exception(f'Invalid method: {method}')
        
    images = iter(dataset.gray)
    for _ in range(25):
        next(images)
    
    global_pose = np.eye(4)
    # trajectory = []
    
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()

    traj_x, traj_z = [], []

    line, = ax.plot([], [], 'g-')  # trajectory line
    point, = ax.plot([], [], 'ro')  # current position

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Trajectory")
    ax.axis("equal")
    
    for i, img in enumerate(images):
        pose = motion_estimator.estimate(img)
        if motion_estimator.returns_global:
            global_pose = pose
            print(pose[:3, 3])
            print()
        else:
            global_pose = global_pose @ pose

        pos = global_pose[:3, 3]

        traj_x.append(pos[0])
        traj_z.append(pos[2])

        # Update plot data
        line.set_data(traj_x, traj_z)
        point.set_data([pos[0]], [pos[2]])

        # Auto-rescale view
        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)
        
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if motion_estimator.matches is not None:
            for m in motion_estimator.matches:
                pt = motion_estimator.prev_kp[m.queryIdx].pt
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
        cv2.imshow('cam0', vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # input('...')
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual odometry.')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence from KITTTI dataset')
    parser.add_argument('--method', type=str, default='eightpoint', help='Pose estimation method')
    args = parser.parse_args()
    main(args.sequence, args.method)