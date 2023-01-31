import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import torch
import os

from common.pose_utils import WorldCube, dump_trajectory_to_tum
from common.pose import Pose
from common.settings import Settings
from common.signals import Slot, Signal, StopSignal
from common.frame import Frame
from common.pose_utils import tensor_to_transform


"""
Listens to data on the given signals, and creates matplotlib plots. 
"""


class DefaultLogger:
    def __init__(self, frame_signal: Signal, keyframe_update_signal: Signal, 
                 world_cube: WorldCube, calibration: Settings, log_directory: str):
        self._world_cube = world_cube
        
        self._frame_slot = frame_signal.register()
        self._keyframe_update_slot = keyframe_update_signal.register()
        
        self._timestamps = []
        self._tracked_path = torch.tensor([])
        self._gt_path = torch.tensor([])

        self._done = False
        self._gt_pose_offset = None

        self._calibration = calibration    

        self._log_directory = log_directory    

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

            if self._gt_pose_offset is None:
                start_pose = frame._gt_lidar_end_pose
                self._gt_pose_offset = start_pose.inv()

            new_pose = frame.get_start_lidar_pose().get_transformation_matrix().detach()

            gt_pose = (self._gt_pose_offset * frame._gt_lidar_end_pose).get_transformation_matrix().detach()

            self._tracked_path = torch.cat([self._tracked_path, new_pose.unsqueeze(0)])
            self._gt_path = torch.cat([self._gt_path, gt_pose.unsqueeze(0)])

            self._timestamps.append(frame.get_start_time())

            if self._tracked_path.shape[0] < 2:
                self._optimized_path = new_pose.unsqueeze(0)
            else:
                relative_transform = self._tracked_path[-2].inverse() @  self._tracked_path[-1]
                self._optimized_path = torch.cat([self._optimized_path, (self._optimized_path[-1] @ relative_transform).unsqueeze(0)])


        while self._keyframe_update_slot.has_value():
            print("Got KeyFrame Update")

            keyframe_state = self._keyframe_update_slot.get_value()

            keyframe_timestamps = torch.tensor([kf["timestamp"] for kf in keyframe_state]).reshape(-1, 1)

            pose_timestamps = torch.tensor(self._timestamps)

            tiled_pose_timestamps = pose_timestamps.expand(len(keyframe_timestamps), -1)
            
            pose_kf_indices = torch.argmin(torch.abs(tiled_pose_timestamps - keyframe_timestamps))
            
            # List of lists, where each sublist i is the index of poses [kf_i, kf_{i} + 1, kf_{i} + 2, ...].
            # Basically, index of poses starting with a KF and including every pose until (not including) the next KF
            idx_chunks = list(torch.tensor_split(torch.arange(len(self._timestamps)), pose_kf_indices.reshape(1,)))
            idx_chunks = [c for c in idx_chunks if len(c) > 0]
            
            self._optimized_path = torch.zeros_like(self._tracked_path)

            for chunk_idx, chunk in enumerate(idx_chunks):
                if len(chunk) == 0:
                    continue
                
                poses = self._tracked_path[chunk[0]:chunk[-1]+1]
                relative_poses = poses @ torch.linalg.inv(poses[0])

                reference_pose_tensor = keyframe_state[chunk_idx]["start_lidar_pose"]
                reference_pose = tensor_to_transform(reference_pose_tensor)
                corrected_poses = reference_pose @ relative_poses
                self._optimized_path[chunk[0]:chunk[-1]+1] = corrected_poses

    def finish(self):
        self.update()

        # Dump it all to TUM format
        os.makedirs(f"{self._log_directory}/trajectory", exist_ok=True)
        pose_timestamps = torch.tensor(self._timestamps)
        dump_trajectory_to_tum(self._tracked_path, pose_timestamps, f"{self._log_directory}/trajectory/tracked.txt")
        dump_trajectory_to_tum(self._optimized_path, pose_timestamps, f"{self._log_directory}/trajectory/optimized.txt")

        # translations = self._tracked_path[:,:3,3]
        # gt_translations = self._gt_path[:,:3,3]

        # xs = [p[0] for p in translations]
        # ys = [p[1] for p in translations]
        # zs = [p[2] for p in translations]

        # xs_gt = [p[0] for p in gt_translations]
        # ys_gt = [p[1] for p in gt_translations]
        # zs_gt = [p[2] for p in gt_translations]

        # xs = np.array(xs)
        # ys = np.array(ys)
        # zs = np.array(zs)
        # xs_gt = np.array(xs_gt)
        # ys_gt = np.array(ys_gt)
        # zs_gt = np.array(zs_gt)

        # data = np.vstack((xs, ys, zs))
        # gt = np.vstack((xs_gt, ys_gt, zs_gt))
        # np.save("data", data)
        # np.save("gt", gt)

        # estimated = np.vstack((xs, ys)).transpose()
        # truth = np.vstack((xs_gt, ys_gt)).transpose()

        # rmse = np.sqrt(np.mean(np.linalg.norm(estimated-truth, axis=1)**2))

        # dists = np.linalg.norm(estimated-truth, axis=1)

        # fig = plt.figure()

        # ax=fig.add_subplot(projection='3d')
        # plt.plot(xs, ys, zs, color='r', label="Estimated")
        # plt.plot(xs_gt, ys_gt, zs_gt, color='g', label="Ground Truth")

        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.title("Ego Vehicle Path")

        # # plt.ylim(-1.0, 1.0)
        # plt.legend()
        # plt.savefig("trajectory_tracking.png", bbox_inches="tight")
        # plt.clf()

        # plt.plot(xs, ys, label="Estimated")
        # plt.plot(xs_gt, ys_gt, color='g', label="Ground Truth")
        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.title("Ego Vehicle Path: Looped Dataset")
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        # # plt.ylim(-1, 1)
        # plt.legend()
        # plt.savefig("ground_truth.png", bbox_inches="tight")

        # self._done = True
