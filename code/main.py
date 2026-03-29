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
        
    images = iter(dataset.gray)
    
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
        
        cv2.imshow('cam0', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual odometry.')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence from KITTTI dataset')
    parser.add_argument('--method', type=str, default='eightpoint', help='Pose estimation method')
    args = parser.parse_args()
    main(args.sequence, args.method)