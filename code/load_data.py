"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2

import pykitti

# Change this to the directory where you store KITTI data
basedir = 'C:/Users/Logan/Documents/Datasets/odom_gray'


def load_dataset(sequence):
    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.odometry(basedir, sequence)
    return dataset

    # dataset.calib:      Calibration data are accessible as a named tuple
    # dataset.timestamps: Timestamps are parsed into a list of timedelta objects
    # dataset.poses:      List of ground truth poses T_w_cam0
    # dataset.camN:       Generator to load individual images from camera N
    # dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)

    # Grab some data
    first_gray = next(iter(dataset.gray))
    # first_cam1 = next(iter(dataset.cam1))

    # Display some of the data
    # np.set_printoptions(precision=4, suppress=True)
    # print('\nSequence: ' + str(dataset.sequence))
    # print('\nFrame range: ' + str(dataset.frames))

    # print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))

    # f, ax = plt.subplots(2, 2, figsize=(15, 5))
    # ax[0, 0].imshow(first_gray[0], cmap='gray')
    # ax[0, 0].set_title('Left Gray Image (cam0)')

    # ax[0, 1].imshow(first_cam1, cmap='gray')
    # ax[0, 1].set_title('Right Gray Image (cam1)')

    # Extract positions from all poses
    
def plot_poses(dataset):
    positions = np.array([pose[:3, 3] for pose in dataset.poses])

    # 2D trajectory (top-down view: x vs z)
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 2])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Trajectory (Top View)')
    plt.axis('equal')
    plt.grid()

    # 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')

    plt.show()
    
def play_video(dataset):
    for i, (left_img, _) in enumerate(dataset.gray):
        left_img = np.array(left_img)
        cv2.imshow('cam0', left_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    dataset = load_dataset('00')
    play_video(dataset)
    plot_poses(dataset)