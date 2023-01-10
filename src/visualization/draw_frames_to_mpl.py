import numpy as np
from common.signals import Slot, Signal, StopSignal
from common.frame import Frame
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from common.pose_utils import WorldCube
from common.pose import Pose
from common.settings import Settings
import torch
import cv2

"""
Listens to data on the given signals, and creates matplotlib plots. 
"""


class MplFrameDrawer:
    def __init__(self, frame_signal: Signal, world_cube: WorldCube, calibration: Settings):
        self._world_cube = world_cube
        self._frame_slot = frame_signal.register()

        self._path = []
        self._gt_path = []

        self._done = False
        self._gt_pose_offset = None

        self._calibration = calibration        

    def update(self):
        if self._done:
            while self._frame_slot.has_value():
                self._frame_slot.get_value()
            return

        while self._frame_slot.has_value():
            frame: Frame = self._frame_slot.get_value()
            if isinstance(frame, StopSignal):
                self._done = True
                break

            print(f"Frame goes {frame.lidar_points.get_start_time()} -> {frame.lidar_points.get_end_time()}")

            if self._gt_pose_offset is None:
                start_pose = frame._gt_lidar_end_pose
                print("Start Pose:", start_pose.get_transformation_matrix())
                self._gt_pose_offset = start_pose.inv()

            new_pose = frame.get_end_lidar_pose()
            new_pose = new_pose.get_translation()

            gt_pose = self._gt_pose_offset * frame._gt_lidar_end_pose
            # print("lidar gt", frame._gt_lidar_end_pose.get_transformation_matrix())
            # print("gt offset", self._gt_pose_offset.get_transformation_matrix())
            # print("lidar gt transformed", gt_pose.get_transformation_matrix())

            gt_pose = gt_pose.get_translation()

            self._path.append(new_pose)
            self._gt_path.append(gt_pose)

            # image = frame.start_image.image.detach().cpu().numpy() * 255

            # points = frame.get_end_lidar_pose().get_translation().reshape((3,1)) + frame.lidar_points.ray_directions * frame.lidar_points.distances
            # points = points.detach().cpu().numpy()

            # rotmat = frame.get_start_camera_pose().get_rotation().detach().cpu().numpy()
            # rotvec = cv2.Rodrigues(rotmat)[0]
            # transvec = frame.get_start_camera_pose().get_translation().detach().cpu().numpy()
            # K = self._calibration.camera_intrinsic.k.detach().cpu().numpy()
            # d = self._calibration.camera_intrinsic.distortion.detach().cpu().numpy()
            # im_pts = cv2.projectPoints(points, rotvec, transvec, K, d)[0]

            # for i in range(im_pts.shape[0]):
            #     pt = im_pts[i][0].astype(int)
            #     pt[0], pt[1] = pt[1], pt[0]
            #     if not np.all(pt > 0):
            #         continue
            #     if pt[0] >= image.shape[1] or pt[1] >= image.shape[0]:
            #         continue
            #     image = cv2.circle(image, pt, 1, (0,0,255), 2)

            # cv2.imwrite("test_image.png", image)            
            

    def finish(self):
        self.update()

        xs = [p.detach()[0] for p in self._path]
        ys = [p.detach()[1] for p in self._path]
        zs = [p.detach()[2] for p in self._path]

        xs_gt = [p.detach()[0] for p in self._gt_path]
        ys_gt = [p.detach()[1] for p in self._gt_path]
        zs_gt = [p.detach()[2] for p in self._gt_path]

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        xs_gt = np.array(xs_gt)
        ys_gt = np.array(ys_gt)
        zs_gt = np.array(zs_gt)

        data = np.vstack((xs, ys, zs))
        gt = np.vstack((xs_gt, ys_gt, zs_gt))
        np.save("data", data)
        np.save("gt", gt)

        estimated = np.vstack((xs, ys)).transpose()
        truth = np.vstack((xs_gt, ys_gt)).transpose()

        rmse = np.sqrt(np.mean(np.linalg.norm(estimated-truth, axis=1)**2))

        dists = np.linalg.norm(estimated-truth, axis=1)

        fig = plt.figure()

        ax=fig.add_subplot(projection='3d')
        plt.plot(xs, ys, zs, color='r', label="Estimated")
        plt.plot(xs_gt, ys_gt, zs_gt, color='g', label="Ground Truth")

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Ego Vehicle Path")

        # plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.savefig("trajectory_tracking.png", bbox_inches="tight")
        plt.clf()

        plt.plot(xs, ys, label="Estimated")
        plt.plot(xs_gt, ys_gt, color='g', label="Ground Truth")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Ego Vehicle Path: Looped Dataset")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # plt.ylim(-1, 1)
        plt.legend()
        plt.savefig("ground_truth.png", bbox_inches="tight")

        self._done = True
