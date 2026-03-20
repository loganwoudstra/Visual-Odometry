# reference: https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_odometry.py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pykitti

basedir = 'C:/Users/Logan/Documents/Datasets/odom_gray'

class Dataset:
    def __init__(self, sequence) -> None:
        # dataset.calib:      Calibration data are accessible as a named tuple
        # dataset.timestamps: Timestamps are parsed into a list of timedelta objects
        # dataset.poses:      List of ground truth poses T_w_cam0
        # dataset.camN:       Generator to load individual images from camera N
        # dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
        self.dataset = pykitti.odometry(basedir, sequence)
        self.poses = self.dataset.poses
        self.K = self.dataset.calib.K_cam0
        
    @property
    def gray(self):
        """original dataset returns img as PIL"""
        for img in self.dataset.cam0:
            yield np.array(img)
    
    def plot_poses(self):
        positions = np.array([pose[:3, 3] for pose in self.poses])

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
    
    def play_video(self):
        for img in self.gray:
            cv2.imshow('cam0', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    dataset = Dataset('00')
    dataset.plot_poses()
    dataset.play_video()
    