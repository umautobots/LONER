import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import torch
import os
from typing import Union

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
        
        self._timestamps = torch.Tensor([])

        # ICP Only
        self._estimated_trajectory = torch.Tensor([])

        self._gt_path = torch.Tensor([])
        
        # Result of propagating tracking to the most recent keyframe
        self._frame_log = torch.Tensor([])

        self._frame_done = False
        self._keyframe_done = False
        self._gt_pose_offset = None

        self._calibration = calibration    

        self._log_directory = log_directory

        self._t_world_to_kf = torch.eye(4)
        self._t_kf_to_frame = torch.eye(4)

    def update(self):        
        if self._frame_done:
            while self._frame_slot.has_value():
                self._frame_slot.get_value()

        while self._frame_slot.has_value():
            frame: Frame = self._frame_slot.get_value()
            if isinstance(frame, StopSignal):
                self._frame_done = True
                break

            frame = frame.clone().to('cpu')

            if self._gt_pose_offset is None:
                start_pose = frame._gt_lidar_pose
                self._gt_pose_offset = start_pose.inv()

            tracked_pose = frame.get_lidar_pose().get_transformation_matrix().detach().cpu()
            gt_pose_raw = frame._gt_lidar_pose
            frame_time = frame.get_time()

            gt_pose = (self._gt_pose_offset * gt_pose_raw).get_transformation_matrix().detach()

            self._estimated_trajectory = torch.cat([self._estimated_trajectory, tracked_pose.unsqueeze(0)])
            self._gt_path = torch.cat([self._gt_path, gt_pose.unsqueeze(0)])

            self._timestamps = torch.cat([self._timestamps, torch.tensor([frame_time])])
            
            if len(self._estimated_trajectory) > 1:
                relative_transform = self._estimated_trajectory[-2].inverse() @  self._estimated_trajectory[-1]
            else:
                relative_transform = tracked_pose

            self._t_kf_to_frame = self._t_kf_to_frame @ relative_transform

            t_world_to_frame_opt = self._t_world_to_kf @ self._t_kf_to_frame
            
            self._frame_log = torch.cat([self._frame_log, t_world_to_frame_opt.unsqueeze(0)])

        while self._keyframe_update_slot.has_value():
            keyframe_state = self._keyframe_update_slot.get_value()
          
            if isinstance(keyframe_state, StopSignal):
                self._frame_done = True
                break
            
            self._last_recv_keyframe_state = keyframe_state
            
            most_recent_kf = keyframe_state[-1]

            kf_time = most_recent_kf["timestamp"]
            kf_pose_tensor = most_recent_kf["lidar_pose"]

            kf_idx = torch.argmin(torch.abs(self._timestamps - kf_time)).item()

            self._t_world_to_kf = Pose(pose_tensor=kf_pose_tensor).get_transformation_matrix()

            tracked_pose = self._estimated_trajectory[kf_idx]
            most_recent_tracked_pose = self._estimated_trajectory[-1]
            
            self._t_kf_to_frame = tracked_pose.inverse() @ most_recent_tracked_pose

    def finish(self):
        self.update()

        keyframe_timestamps = torch.tensor([kf["timestamp"] for kf in self._last_recv_keyframe_state])

        keyframe_trajectory = torch.stack([Pose(pose_tensor=kf["lidar_pose"]).get_transformation_matrix() for kf in self._last_recv_keyframe_state])

        # Which kf each pose is closest (temporally) to
        pose_kf_indices = torch.bucketize(self._timestamps, keyframe_timestamps, right=False).long() - 1
        pose_kf_indices[pose_kf_indices < 0] = 0
        
        # Dump it all to TUM format
        os.makedirs(f"{self._log_directory}/trajectory", exist_ok=True)
        dump_trajectory_to_tum(self._estimated_trajectory, self._timestamps, f"{self._log_directory}/trajectory/tracking_only.txt")
        dump_trajectory_to_tum(self._frame_log, self._timestamps, f"{self._log_directory}/trajectory/online_estimates.txt")
        dump_trajectory_to_tum(keyframe_trajectory, keyframe_timestamps, f"{self._log_directory}/trajectory/keyframe_trajectory.txt")
