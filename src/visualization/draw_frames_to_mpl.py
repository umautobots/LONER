import numpy as np
from common.signals import Slot, Signal
from common.frame import Frame
from scipy.spatial.transform import Rotation as R, Slerp
from common.utils import StopSignal
import matplotlib.pyplot as plt
from common.pose_utils import WorldCube
import torch


class MplFrameDrawer:
    def __init__(self, frame_signal: Signal, world_cube: WorldCube):
        self._world_cube = world_cube
        self._frame_slot = frame_signal.register()

        self._path = []
        self._gt_path = []

        self._done = False
        self._gt_pose_offset = None

    def update(self):
        if self._done:
            while self._frame_slot.has_value():
                self._frame_slot.get_value()
            return

        while self._frame_slot.has_value():
            frame = self._frame_slot.get_value()

            if isinstance(frame, StopSignal):
                self._done = True
                break

            if self._gt_pose_offset is None:
                start_pose = frame._gt_lidar_start_pose
                self._gt_pose_offset = start_pose.inv()

            new_pose = frame.get_start_lidar_pose()
            new_pose = new_pose.transform_world_cube(self._world_cube, reverse=True, ignore_shift=True)
            new_pose = new_pose.get_transformation_matrix()[:3, 3]

            gt_pose = self._gt_pose_offset * frame._gt_lidar_start_pose
            gt_pose = gt_pose.get_transformation_matrix()[:3, 3]    
                

            self._path.append(new_pose)
            self._gt_path.append(gt_pose)

    def finish(self):
        self.update()

        xs = [p.detach()[0] for p in self._path]
        ys = [p.detach()[1] for p in self._path]

        xs_gt = [p.detach()[0] for p in self._gt_path]
        ys_gt = [p.detach()[1] for p in self._gt_path]

        xs = np.array(xs)
        ys = np.array(ys)
        xs_gt = np.array(xs_gt)
        ys_gt = np.array(ys_gt)

        estimated = np.vstack((xs, ys)).transpose()
        truth = np.vstack((xs_gt, ys_gt)).transpose()

        rmse = np.sqrt(np.mean(np.linalg.norm(estimated-truth, axis=1)**2))

        dists = np.linalg.norm(estimated-truth, axis=1)
        plt.plot(xs, ys, color='r', label="Estimated")
        plt.plot(xs_gt, ys_gt, color='g', label="Ground Truth")

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Ego Vehicle Path")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.savefig("trajectory_tracking.png", bbox_inches="tight")
        plt.clf()

        # plt.plot(xs_gt, ys_gt, color='g', label="Ground Truth")
        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.title("Ego Vehicle Path: Looped Dataset")
        # ax = plt.gca()
        # # ax.set_aspect('equal', adjustable='box')
        # # plt.ylim(-.125, .125)
        # plt.legend()
        # plt.savefig("ground_truth.png", bbox_inches="tight")

        
        self._done = True
