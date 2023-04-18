import torch
from typing import Tuple, Union

from common.pose_utils import WorldCube
from common.pose import Pose
import pytorch3d.transforms as transf

NUMERIC_TOLERANCE = 1e-9

class Image:
    """ Image class for holding images.

    A simple wrapper containing an image and a timestamp
    """

    ## Constructor
    # @param image: a torch Tensor of RGB or Binary data
    # @param timestamp: the time at which the image was captured
    def __init__(self, image: torch.Tensor, timestamp: float):
        self.image = image
        self.timestamp = timestamp

        self.shape = self.image.shape

    ## @returns a copy of the current image
    def clone(self) -> "Image":
        if isinstance(self.timestamp, torch.Tensor):
            new_ts = self.timestamp.clone()
        else:
            new_ts = self.timestamp
        return Image(self.image.clone(), new_ts)

    ## Moves all items in the image to the specified device, in-place. Also returns the current image.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "Image":
        self.image = self.image.to(device)

        if isinstance(self.timestamp, torch.Tensor):
            self.timestamp = self.timestamp.to(device)
        else:
            self.timestamp = torch.Tensor([self.timestamp]).to(device)
        return self


class LidarScan:
    """ LidarScan class for handling lidar data

    Represents Lidar data as ray directions, ray origin offsets, and timestamps.
    Note that this intentionally does not store the location of the pose. To reconstruct
    a point cloud, for each ray you would do the following, given a pose of lidar
    at time t of $$T_{l,t}$$:

    $$point = T_{l,timestamps[i]} + ray_directions[i] * distances[i]$$
    """

    ## Constructor
    # @param ray_directions: Direction of each ray. 3xn tensor.
    # @param distances: Distance of each ray. 1xn tensor
    # @param timestamps: The time at which each laser fired. Used for motion compensation.
    # @precond: timestamps are sorted. You will have mysterious problems if this isn't true.
    def __init__(self,
                 ray_directions: torch.Tensor = torch.Tensor(),
                 distances: torch.Tensor = torch.Tensor(),
                 timestamps: torch.Tensor = torch.Tensor(),
                 sky_rays: torch.Tensor = None) -> None:

        self.ray_directions = ray_directions
        self.distances = distances
        self.timestamps = timestamps
        self.sky_rays = sky_rays

    ## @returns the number of points in the scan
    def __len__(self) -> int:
        return self.timestamps.shape[0]

    ## Gets the timestamp of the first lidar point
    def get_start_time(self) -> torch.Tensor:
        return self.timestamps[0]

    ## Gets the timestamp of the last lidar point
    def get_end_time(self) -> torch.Tensor:
        return self.timestamps[-1]

    ## Removes all points from the current scan. Also @returns self
    def clear(self) -> "LidarScan":
        self.ray_directions = torch.Tensor()
        self.distances = torch.Tensor()
        self.timestamps = torch.Tensor()
        if self.sky_rays is not None:
            self.sky_rays = torch.Tensor()
            
        return self

    ## @returns a deep copy of the current scan
    def clone(self) -> "LidarScan":
        return LidarScan(self.ray_directions.clone(),
                         self.distances.clone(),
                         self.timestamps.clone(),
                         self.sky_rays.clone() if self.sky_rays is not None else None)

    ## Removes the first @p num_points points from the scan. Also @returns self.
    def remove_points(self, num_points: int) -> "LidarScan":
        self.ray_directions = self.ray_directions[..., num_points:]
        self.distances = self.distances[num_points:]
        self.timestamps = self.timestamps[num_points:]

        return self

    ## Copies points from the @p other scan into this one. Also returns self.
    def merge(self, other: "LidarScan") -> "LidarScan":
        self.add_points(other.ray_directions,
                        other.distances,
                        other.timestamps,
                        other.sky_rays)
        return self

    ## Moves all items in the LidarScan to the specified device, in-place.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    # @returns the current scan.
    def to(self, device: Union[int, str]) -> "LidarScan":
        self.ray_directions = self.ray_directions.to(device)
        self.distances = self.distances.to(device)
        self.timestamps = self.timestamps.to(device)
        return self

    ## Add points to the current scan, with same arguments as constructor. @returns self.
    def add_points(self,
                   ray_directions: torch.Tensor,
                   distances: torch.Tensor,
                   timestamps: torch.Tensor,
                   sky_rays: torch.Tensor = None) -> "LidarScan":

        if self.ray_directions.shape[0] == 0:
            self.distances = distances
            self.ray_directions = ray_directions
            self.timestamps = timestamps
        else:
            self.ray_directions = torch.cat(
                (self.ray_directions, ray_directions), dim=-1)
            self.timestamps = torch.cat((self.timestamps, timestamps), dim=-1)
            self.distances = torch.cat((self.distances, distances), dim=-1)

        if sky_rays is not None:
            if self.sky_rays is None:
                self.sky_rays = sky_rays
            else:
                self.sky_rays = torch.cat((self.sky_rays, sky_rays), dim=-1)
        return self

    def get_sky_scan(self, distance: float) -> "LidarScan":
        sky_dirs = self.sky_rays
        distances = torch.full_like(sky_dirs[0], distance)
        times = torch.full_like(sky_dirs[0], self.timestamps[-1])

        return LidarScan(sky_dirs, distances, times)

    ## Given a start and end poses, applies motion compensation to the lidar points.
    # This first projects points into the global frame using the start and end poses,
    # then projects the result back into the target frame.
    # @param poses: A start and end pose
    # @param timestamps: Timestamps corresponding to each of the provided poses
    # @param target_frame: What frame to motion compensate into.
    # @param: use_gpu: If true, will do the compensation on the GPU (slightly faster)
    def motion_compensate(self,
                          poses: Tuple[Pose, Pose], 
                          timestamps: Tuple[float, float],
                          target_frame: Pose,
                          use_gpu: bool = False):
        
        device = 'cuda' if use_gpu else 'cpu'

        start_pose, end_pose = poses
        start_ts, end_ts = timestamps

        N = self.timestamps.shape[0]
        interp_factors = ((self.timestamps - start_ts)/(end_ts - start_ts)).to(device)

        start_trans, end_trans = start_pose.get_translation().to(device), end_pose.get_translation().to(device)
        delta_translation = end_trans - start_trans

        output_translations = delta_translation * interp_factors[:, None] + start_trans

        start_rot = start_pose.get_transformation_matrix()[:3, :3]
        end_rot = end_pose.get_transformation_matrix()[:3, :3]

        relative_rotation = torch.linalg.inv(start_rot) @ end_rot

        rotation_axis_angle = transf.matrix_to_axis_angle(relative_rotation).to(device)

        rotation_angle = torch.linalg.norm(rotation_axis_angle).to(device)

        if rotation_angle < NUMERIC_TOLERANCE:
            rotation_matrices = torch.eye(3, device=device).repeat(N, 1, 1)
        else:
            rotation_axis = rotation_axis_angle / rotation_angle

            rotation_amounts = rotation_angle * interp_factors[:, None]
            output_rotation_axis_angles = rotation_amounts * rotation_axis

            rotation_matrices = transf.axis_angle_to_matrix(
                output_rotation_axis_angles)

        rotation_matrices = start_rot.to(device) @ rotation_matrices
        
        T_world_to_compensated_lidar = torch.cat(
            [rotation_matrices, output_translations.unsqueeze(2)], dim=-1)
        h = torch.Tensor([0, 0, 0, 1]).to(T_world_to_compensated_lidar.device).repeat(N, 1, 1)
        T_world_to_compensated_lidar = torch.cat([T_world_to_compensated_lidar, h], dim=1)


        T_world_to_target = target_frame.get_transformation_matrix().detach().to(device)
        T_target_to_compensated_lidar = torch.linalg.inv(T_world_to_target) @ T_world_to_compensated_lidar

        points_lidar = self.ray_directions*self.distances
        points_lidar_homog = torch.vstack((points_lidar, torch.ones_like(points_lidar[0]))).to(device)
        motion_compensated_points = (T_target_to_compensated_lidar @ points_lidar_homog.T.unsqueeze(2)).squeeze(2).T[:3]

        self.distances = torch.linalg.norm(motion_compensated_points, dim=0)
        self.ray_directions = (motion_compensated_points / self.distances).to(self.timestamps.device)
        self.distances = self.distances.to(self.timestamps.device)