from dataset import Dataset
from feature_tracker import FeatureTracker
from motion_estimator import MotionEstimator
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(sequence):
    dataset = Dataset(sequence)
    tracker = FeatureTracker()
    
    # K = dataset.
    motion_estimator = MotionEstimator(dataset.K)
    
    images = iter(dataset.gray)
    img_prev = next(images)
    kp_des_prev = tracker.detect(img_prev)
    
    cv2.imshow('cam0', img_prev)
    
    pose = np.eye(4)
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
        kp_des = tracker.detect(img)
        matches = tracker.match(kp_des_prev, kp_des)
        pts1, pts2 = tracker.point_correspondences(kp_des_prev[0], kp_des[0], matches)
        
        motion = motion_estimator.estimate(pts1, pts2, ransac=True)
        # pose = pose @ motion
        pose = pose @ np.linalg.inv(motion)
        pos = pose[:3, 3]

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

        img_prev = img
        kp_des_prev = kp_des
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual odometry.')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence from KITTTI dataset')
    args = parser.parse_args()
    main(args.sequence)