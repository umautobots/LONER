import cProfile
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
from common.ray_utils import CameraRayDirections, LidarRayDirections
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
                         ignore_world_cube: bool = False,
                         sky_indices: torch.Tensor = None) -> torch.Tensor:

        lidar_scan = self.get_lidar_scan()

        # N x 4 x 4
        if use_gt_poses:
            lidar_poses = self._frame._gt_lidar_pose.get_transformation_matrix()
        else:
            lidar_poses = self._frame.get_lidar_pose().get_transformation_matrix()

        if sky_indices is None:
            ray_dirs = LidarRayDirections(lidar_scan)
            lidar_rays, lidar_depths = ray_dirs.build_lidar_rays(lidar_indices, ray_range, world_cube, lidar_poses, ignore_world_cube)
        
        else:
            sky_scan = lidar_scan.get_sky_scan(ray_range[1] + 1)
            sky_dirs = LidarRayDirections(sky_scan)
            sky_rays, sky_depths = sky_dirs.build_lidar_rays(sky_indices, ray_range, world_cube, lidar_poses.detach(), ignore_world_cube)
            
            ray_dirs = LidarRayDirections(lidar_scan)
            lidar_rays, lidar_depths = ray_dirs.build_lidar_rays(lidar_indices, ray_range, world_cube, lidar_poses, ignore_world_cube)

            lidar_rays = torch.cat((lidar_rays, sky_rays))
            lidar_depths = torch.cat((lidar_depths, sky_depths))
        return lidar_rays, lidar_depths
        
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
            "timestamp": self.get_time().detach().cpu().clone(),
            "lidar_to_camera": self._frame._lidar_to_camera.get_pose_tensor().detach().cpu().clone(),
            "lidar_pose":  self._frame.get_lidar_pose().get_pose_tensor().detach().cpu().clone(),
            "gt_lidar_pose": self._frame._gt_lidar_pose.get_pose_tensor().detach().cpu().clone(),
            "tracked_pose": self._tracked_lidar_pose.get_pose_tensor().detach().cpu().clone()
        }

        return state_dict