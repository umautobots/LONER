from typing import Tuple, Union
from enum import Enum

import pytorch3d.transforms as transf
import torch
import numpy as np
from pathlib import Path
import cv2
import os

from common.frame import Frame
from common.pose import Pose
from common.pose_utils import WorldCube
from common.ray_utils import CameraRayDirections, get_far_val
from common.sensors import Image, LidarScan

NUMERIC_TOLERANCE = 1e-9

class KeyFrame:
    """ The KeyFrame class stores a frame an additional metadata to be used in optimization.
    """

    ## Constructor: Create a KeyFrame from input Frame @p frame.
    def __init__(self, frame: Frame, device: int = None) -> None:


        self._frame = frame.to(device)

        self._device = device

        self._tracked_lidar_pose: Pose = frame.get_lidar_pose().clone()

        self.is_anchored = False

        self.rgb_loss_distribution = None
        self.lidar_loss_distribution = None

        self.lidar_buckets = None

    def to(self, device) -> "KeyFrame":
        self._frame.to(device)
        self._device = device
        return self

    def detach(self) -> "KeyFrame":
        self._frame.detach()
        return self

    def __str__(self) -> str:
        return str(self._frame)

    def __repr__(self) -> str:
        return str(self)

    def get_camera_pose(self) -> Pose:
        return self._frame.get_camera_pose()

    def get_lidar_pose(self) -> Pose:
        return self._frame.get_lidar_pose()

    def get_lidar_scan(self) -> LidarScan:
        return self._frame.lidar_points

    def get_time(self) -> float:
        return self._frame.get_time()


    ## For all the points in the frame, create lidar rays in the format Loner wants
    def build_lidar_rays(self,
                         lidar_indices: torch.Tensor,
                         ray_range: torch.Tensor,
                         world_cube: WorldCube,
                         use_gt_poses: bool = False,
                         ignore_world_cube: bool = False) -> torch.Tensor:

        lidar_scan = self.get_lidar_scan()

        depths = lidar_scan.distances[lidar_indices] / world_cube.scale_factor
        directions = lidar_scan.ray_directions[:, lidar_indices]
        timestamps = lidar_scan.timestamps[lidar_indices]

        # N x 4 x 4
        if use_gt_poses:
            lidar_poses = self._frame._gt_lidar_pose.get_transformation_matrix()
        else:
            lidar_poses = self._frame.get_lidar_pose().get_transformation_matrix()

        ray_origins: torch.Tensor = lidar_poses[..., :3, 3]
        ray_origins = ray_origins + world_cube.shift
        ray_origins = ray_origins / world_cube.scale_factor

        ray_origins = ray_origins.tile(len(timestamps), 1)

        # N x 3 x 3 (N homogenous transformation matrices)
        lidar_rotations = lidar_poses[..., :3, :3]
        
        # N x 3 x 1. This takes a 3xN matrix and makes it 1x3xN, then Nx3x1
        directions_3d = directions.unsqueeze(0).swapaxes(0, 2)

        # rotate ray directions from sensor coordinates to world coordinates
        ray_directions = lidar_rotations @ directions_3d

        # ray_directions is now Nx3x1, we want Nx3.
        ray_directions = ray_directions.squeeze()

        # Note to self: don't use /= here. Breaks autograd.
        ray_directions = ray_directions / \
            torch.norm(ray_directions, dim=1, keepdim=True)

        view_directions = -ray_directions

        if not ignore_world_cube:
            assert (ray_origins.abs().max(dim=1)[0] > 1).sum() == 0, \
                f"{(ray_origins.abs().max(dim=1)[0] > 1).sum()//3} ray origins are outside the world cube"

        near = ray_range[0] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])
        far_range = ray_range[1] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])

        far_clip = get_far_val(ray_origins, ray_directions, no_nan=True)
        far = torch.minimum(far_range, far_clip)

        rays = torch.cat([ray_origins, ray_directions, view_directions,
                          torch.zeros_like(ray_origins[:, :2]),
                          near, far], 1)
                          
        # Only rays that have more than 1m inside world
        if ignore_world_cube:
            return rays, depths
        else:
            valid_idxs = (far > (near + 1. / world_cube.scale_factor))[..., 0]
            return rays[valid_idxs], depths[valid_idxs]
    
    ## Given the images, create camera rays in Loner's format
    def build_camera_rays(self,
                          first_camera_indices: torch.Tensor,
                          ray_range: torch.Tensor,
                          cam_ray_directions: CameraRayDirections,
                          world_cube: WorldCube,
                          use_gt_poses: bool = False,
                          detach_rgb_from_poses: bool = False) -> torch.Tensor:


        if use_gt_poses:
            cam_pose = self._frame._gt_lidar_pose * self._frame._lidar_to_camera
        else:
            cam_pose = self._frame.get_camera_pose()
        
        if detach_rgb_from_poses:
            cam_pose = cam_pose.detach()

        rays, intensities = cam_ray_directions.build_rays(first_camera_indices, 
            cam_pose, self._frame.image, world_cube, ray_range)

        return rays, intensities

    def get_pose_state(self) -> dict:
        state_dict = {
            "timestamp": self.get_time(),
            "lidar_to_camera": self._frame._lidar_to_camera.get_pose_tensor().detach().cpu(),
            "lidar_pose":  self._frame.get_lidar_pose().get_pose_tensor().detach().cpu(),
            "gt_lidar_pose": self._frame._gt_lidar_pose.get_pose_tensor().detach().cpu(),
            "tracked_pose": self._tracked_lidar_pose.get_pose_tensor().detach().cpu()
        }

        return state_dict