from dataset import Dataset
from motion_estimation import EightPointEstimator, OpenCVPnpEstimator, OpenCVMatrixEstimator, FivePointEstimator, DLTEstimator
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(sequence, method):
    dataset = Dataset(sequence)
    K = dataset.K

    match method:
        case 'eightpoint':
            motion_estimator = EightPointEstimator(K)
        case 'fivepoint':
            motion_estimator = FivePointEstimator(K)
        case 'dlt':
            motion_estimator = DLTEstimator(K)
        case 'pnp':
            motion_estimator = OpenCVPnpEstimator(K)
        case 'opencv_matrix':
            motion_estimator = OpenCVMatrixEstimator(K)
        case _:
            raise Exception(f'Invalid method: {method}')
        
    images = iter(dataset.gray)
    for _ in range(225):
        next(images)
    
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()


    line, = ax.plot([], [], 'g-')  # trajectory line
    point, = ax.plot([], [], 'ro')  # current position

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Trajectory")
    ax.axis("equal")
    
    for i, img in enumerate(images):
        motion_estimator.step(img)
        
        pos_trajectory = [pose[:3, 3] for pose in motion_estimator.trajectory]
        traj_x = [pos[0] for pos in pos_trajectory]
        traj_z = [pos[2] for pos in pos_trajectory]

        # Update plot data
        line.set_data(traj_x, traj_z)
        curr_pos = (traj_x[-1].item(), traj_z[-1].item())
        point.set_data([curr_pos[0]], [curr_pos[1]])
        print(curr_pos)
        print()

        # Auto-rescale view
        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)
        
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if motion_estimator.matches is not None:
            for m in motion_estimator.matches:
                pt = motion_estimator.kp[m.queryIdx].pt
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