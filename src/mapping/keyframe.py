import torch
from typing import Tuple
import pytorch3d.transforms

from common.frame import Frame
from common.pose import Pose
from common.sensors import LidarScan, Image
from common.pose_utils import WorldCube
from common.ray_utils import get_far_val, CameraRayDirections



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
    
    def __str__(self) -> str:
        return str(self._frame)
    
    def __repr__(self) -> str:
        return str(self)

    def get_start_camera_transform(self) -> torch.Tensor:
        return self._frame.get_start_camera_transform()

    def get_end_camera_transform(self) -> torch.Tensor:
        return self._frame.get_end_camera_transform()

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

    def interpolate_lidar_poses(self, timestamps) -> torch.Tensor:
        assert timestamps.dim() == 1

        N = timestamps.shape[0]

        start_time = self.get_start_time()
        end_time = self.get_end_time()

        interp_factors = (timestamps - start_time)/(end_time - start_time)
        
        start_trans = self.get_start_lidar_pose().get_translation()
        end_trans = self.get_end_lidar_pose().get_translation()
        delta_translation = end_trans - start_trans

        # the interp_factors[:, None] adds a singleton dimension to support element-wise mult
        output_translations = delta_translation * interp_factors[:,None] + start_trans

        # Reshape it from Nx3 to Nx3x1
        output_translations.unsqueeze_(2)

        # Interpolate/extrapolate rotations via axis angle
        start_rot = self.get_start_lidar_pose().get_transformation_matrix()[:3,:3]
        end_rot = self.get_end_lidar_pose().get_transformation_matrix()[:3,:3]

        relative_rotation = torch.linalg.inv(start_rot) @ end_rot

        rotation_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(relative_rotation)

        rotation_angle = torch.linalg.norm(rotation_axis_angle)
        rotation_axis = rotation_axis_angle / rotation_angle

        rotation_amounts = rotation_angle * interp_factors[:,None]
        output_rotation_axis_angles = rotation_amounts * rotation_axis

        rotation_matrices = pytorch3d.transforms.axis_angle_to_matrix()

        output_rotmats = torch.cat([rotation_matrices, output_translations], dim=-1)

        # make it homogenous
        h = torch.Tensor([0,0,0,1]).to(output_rotmats.device).repeat(N, 1, 1)
        output_rotmats_homo = torch.cat([output_rotmats, h], dim=1)

        return output_rotmats_homo

    def build_lidar_rays(self,
                         lidar_indices: torch.Tensor,
                         ray_range: torch.Tensor,
                         world_cube: WorldCube) -> torch.Tensor:

        lidar_scan = self.get_lidar_scan()

        depths = lidar_scan.distances[lidar_indices]
        directions = lidar_scan.ray_directions[lidar_indices]
        timestamps = lidar_scan.timestamps[lidar_indices]

        origin = lidar_scan.ray_origin_offsets
        assert origin.dim() == 2, "Currently there is not support for unique lidar origin offsets"

        lidar_poses = self.interpolate_lidar_poses(timestamps)

        # N x 3 x 3 (N homogenous transformation matrices)
        lidar_rotations = lidar_poses[..., :3, :3]
        # N x 3 x 1. This takes a 3xN matrix and makes it 1x3xN, then Nx3x1
        directions_3d = directions.unsqueeze(0).swapaxes(0, 2)

        # rotate ray directions from sensor coordinates to world coordinates
        # The transpose(1,2) effectively transposes each 3x3 matrix in the nx3x3 tensor
        ray_directions = directions_3d @ lidar_rotations.transpose(1, 2)

        # ray_directions is now Nx3x1, we want Nx3.
        ray_directions = ray_directions.squeeze()
        ray_directions /= torch.norm(ray_directions, dim=1, keepdim=True)
        
        # Nx3
        lidar_origins = lidar_poses[..., :3, 3]
        # Nx4
        lidar_origins_homo = torch.hstack(
            (lidar_origins, torch.ones_like(lidar_origins[:, :1])))
        lidar_origins_homo_3d = lidar_origins_homo.unsqueeze(0).swapaxes(0, 2)
        ray_origins = lidar_origins_homo_3d @ lidar_poses.transpose(1, 2)
        ray_origins = ray_origins.squeeze()[:,:3]

        view_directions = -ray_directions

        assert (ray_origins.abs().max(dim=1)[0] > 1).sum() == 0, \
            f"{(ray_origins.abs().max(dim=1)[0] > 1).sum()} ray origins are outside the world cube"

        near = ray_range[0] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])
        far_range = ray_range[1] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])
        far_clip = get_far_val(ray_origins, ray_directions)
        far = torch.minimum(far_range, far_clip)

        rays = torch.cat([ray_origins, ray_directions, view_directions,
                          torch.inf * torch.ones_like(ray_origins[:, :2]),
                          near, far], 1)
        
        # Only rays that have more than 1m inside world
        valid_idxs = (far > (near + 1. / world_cube.scale_factor))[...,0]
        return rays[valid_idxs]

    def build_camera_rays(self,
                          first_camera_indices: torch.Tensor,
                          second_camera_indices: torch.Tensor,
                          ray_range: torch.Tensor,
                          cam_ray_directions: CameraRayDirections,
                          world_cube: WorldCube) -> torch.Tensor:
        
        def _build_rays(camera_indices, pose):
            
            directions = cam_ray_directions.directions[camera_indices]
            ray_i_grid = cam_ray_directions.i_meshgrid[camera_indices]
            ray_j_grid = cam_ray_directions.j_meshgrid[camera_indices]

            ray_directions = directions @ pose[:3, :3].T
            ray_directions /= torch.norm(ray_directions, dim=-1, keepdim=True)
            ray_directions = ray_directions[:,:3] #TODO: is needed?

            # We assume for cameras that the near plane distances are all zero
            # TODO: Verify
            ray_origins = torch.zeros_like(ray_directions)
            ray_origins_homo = torch.cat([ray_origins, torch.ones_like(ray_origins[:,:1])], dim=-1)
            ray_origins = ray_origins_homo @ pose[:3, :].T

            view_directions = -ray_directions
            near = ray_range[0] / world_cube.scale_factor * torch.ones_like(ray_origins[:,:1])
            far = get_far_val(ray_origins, ray_directions)
            rays = torch.cat([ray_origins, ray_directions, view_directions,
                              ray_i_grid, ray_j_grid, near, far], 1).float()

            return rays

        first_rays = _build_rays(first_camera_indices, self.get_end_camera_transform())
        second_rays = _build_rays(second_camera_indices, self.get_end_camera_transform())

        return torch.vstack((first_rays, second_rays))












