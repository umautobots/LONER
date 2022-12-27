import torch
from typing import Tuple
import pytorch3d.transforms

from common.frame import Frame
from common.pose import Pose
from common.sensors import LidarScan, Image
from common.pose_utils import WorldCube
from common.ray_utils import get_far_val, CameraRayDirections

NUMERIC_TOLERANCE = 1e-9


class KeyFrame:
    """ The KeyFrame class stores a frame an additional metadata to be used in optimization.
    """

    # Constructor: Create a KeyFrame from input Frame @p frame.
    def __init__(self, frame: Frame, device: int = None) -> None:
        self._frame = frame.to(device)

        # How many RGB samples to sample uniformly
        self.num_uniform_rgb_samples = None

        # How many to use the strategy to choose
        self.num_strategy_rgb_samples = None

        # Same for lidar
        self.num_uniform_lidar_samples = None
        self.num_strategy_lidar_samples = None

        self._device = device

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

    def get_start_camera_pose(self) -> Pose:
        return self._frame.get_start_camera_pose()

    def get_end_camera_pose(self) -> Pose:
        return self._frame.get_end_camera_pose()

    def get_start_lidar_pose(self) -> Pose:
        return self._frame.get_start_lidar_pose()

    def get_end_lidar_pose(self) -> Pose:
        return self._frame.get_end_lidar_pose()

    def get_images(self) -> Tuple[Image, Image]:
        return self._frame.start_image, self._frame.end_image

    def get_lidar_scan(self) -> LidarScan:
        return self._frame.lidar_points

    def get_start_time(self) -> float:
        return self._frame.start_image.timestamp

    def get_end_time(self) -> float:
        return self._frame.end_image.timestamp

    # At the given @p timestamps, interpolate/extrapolate and return the lidar poses.
    # @returns For N timestamps, returns a Nx4x4 tensor with all the interpolated/extrapolated transforms
    def interpolate_lidar_poses(self, timestamps: torch.Tensor) -> torch.Tensor:
        assert timestamps.dim() == 1

        N = timestamps.shape[0]

        start_time = self.get_start_time()
        end_time = self.get_end_time()

        interp_factors = (timestamps - start_time)/(end_time - start_time)

        start_trans = self.get_start_lidar_pose().get_translation()
        end_trans = self.get_end_lidar_pose().get_translation()
        delta_translation = end_trans - start_trans

        # the interp_factors[:, None] adds a singleton dimension to support element-wise mult
        output_translations = delta_translation * \
            interp_factors[:, None] + start_trans

        # Reshape it from Nx3 to Nx3x1
        output_translations = output_translations.unsqueeze(2)

        # Interpolate/extrapolate rotations via axis angle
        start_rot = self.get_start_lidar_pose().get_transformation_matrix()[:3, :3]
        end_rot = self.get_end_lidar_pose().get_transformation_matrix()[:3, :3]

        relative_rotation = torch.linalg.inv(start_rot) @ end_rot

        rotation_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(
            relative_rotation)

        rotation_angle = torch.linalg.norm(rotation_axis_angle)

        if rotation_angle < NUMERIC_TOLERANCE:
            rotation_matrices = torch.eye(3).to(
                timestamps.device).repeat(N, 1, 1)

        else:
            rotation_axis = rotation_axis_angle / rotation_angle

            rotation_amounts = rotation_angle * interp_factors[:, None]
            output_rotation_axis_angles = rotation_amounts * rotation_axis

            rotation_matrices = pytorch3d.transforms.axis_angle_to_matrix(
                output_rotation_axis_angles)

        output_transformations = torch.cat(
            [rotation_matrices, output_translations], dim=-1)

        # make it homogenous
        h = torch.Tensor([0, 0, 0, 1]).to(
            output_transformations.device).repeat(N, 1, 1)
        output_transformations_homo = torch.cat(
            [output_transformations, h], dim=1)

        return output_transformations_homo

    # For all the points in the frame, create lidar rays in the format Cloner wants
    def build_lidar_rays(self,
                         lidar_indices: torch.Tensor,
                         ray_range: torch.Tensor,
                         world_cube: WorldCube) -> torch.Tensor:

        lidar_scan = self.get_lidar_scan()


        # TODO: member variables
        rotate_lidar_points_opengl = torch.Tensor([[0, -1, 0],
                                            [0,  0, 1],
                                            [-1, 0, 0]]).to(self._device)
        rotate_lidar_opengl = torch.Tensor([[0, 0, -1],
                                            [-1,  0, 0],
                                            [0, 1, 0]]).to(self._device)

        depths = lidar_scan.distances[lidar_indices]
        directions = lidar_scan.ray_directions[:, lidar_indices]
        directions = rotate_lidar_points_opengl @ directions
        timestamps = lidar_scan.timestamps[lidar_indices]

        # 4 x 4
        origin_offset = lidar_scan.ray_origin_offsets
        assert origin_offset.dim() == 2, "Currently there is not support for unique lidar origin offsets"

        # N x 4 x 4
        lidar_poses = self.interpolate_lidar_poses(timestamps)

        lidar_poses = lidar_poses @ origin_offset

        # N x 3 x 3 (N homogenous transformation matrices)
        lidar_rotations = lidar_poses[..., :3, :3]

        # N x 3 x 1. This takes a 3xN matrix and makes it 1x3xN, then Nx3x1
        directions_3d = directions.unsqueeze(0).swapaxes(0, 2)

        # rotate ray directions from sensor coordinates to world coordinates
        # The transpose(1,2) effectively transposes each 3x3 matrix in the nx3x3 tensor
        # TODO: This is a bit different from how original cloner calculates it. Verify correctness.
        ray_directions = lidar_rotations @ directions_3d

        # ray_directions is now Nx3x1, we want Nx3.
        ray_directions = ray_directions.squeeze()

        # Note to self: don't use /= here. Breaks autograd.
        ray_directions = ray_directions / \
            torch.norm(ray_directions, dim=1, keepdim=True)

        # Nx3
        ray_origins = lidar_poses[..., :3, 3]

        ray_origins =  ray_origins @ rotate_lidar_opengl
        view_directions = -ray_directions

        assert (ray_origins.abs().max(dim=1)[0] > 1).sum() == 0, \
            f"{(ray_origins.abs().max(dim=1)[0] > 1).sum()} ray origins are outside the world cube"

        near = ray_range[0] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])
        far_range = ray_range[1] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])

        # TODO: Does no_nan cause problems here? 
        far_clip = get_far_val(ray_origins, ray_directions, no_nan=True)
        far = torch.minimum(far_range, far_clip)

        rays = torch.cat([ray_origins, ray_directions, view_directions,
                          torch.zeros_like(ray_origins[:, :2]),
                          near, far], 1)
                          
        # Only rays that have more than 1m inside world
        valid_idxs = (far > (near + 1. / world_cube.scale_factor))[..., 0]
        return rays[valid_idxs], depths[valid_idxs]

    # Given the images, create camera rays in Cloner's format
    def build_camera_rays(self,
                          first_camera_indices: torch.Tensor,
                          second_camera_indices: torch.Tensor,
                          ray_range: torch.Tensor,
                          cam_ray_directions: CameraRayDirections,
                          world_cube: WorldCube) -> torch.Tensor:

        first_rays, first_intensities = cam_ray_directions.build_rays(first_camera_indices,
                                self.get_start_camera_pose(),
                                self._frame.start_image, 
                                world_cube,
                                ray_range)

        second_rays, second_intensities = cam_ray_directions.build_rays(second_camera_indices,
                                self.get_end_camera_pose(),
                                self._frame.end_image,
                                world_cube,
                                ray_range)

        return torch.vstack((first_rays, second_rays)), torch.vstack((first_intensities, second_intensities))

    def get_pose_state(self) -> dict:
        return {
            "timestamp": self.get_start_time(),
            "start_lidar_pose": self._frame.get_start_lidar_pose().get_pose_tensor(),
            "end_lidar_pose": self._frame.get_end_lidar_pose().get_pose_tensor(),
            "lidar_to_camera": self._frame._lidar_to_camera.get_pose_tensor()
        }