import struct

import numpy as np
import torch
import open3d as o3d
from kornia.geometry.calibration import undistort_points

from common.pose import Pose
from common.sensors import Image
from common.settings import Settings
from common.pose_utils import WorldCube
from common.sensors import Image, LidarScan

from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
import rosbag
import rospy
import trimesh
import pymesh
from packaging import version
import skimage

# For each dimension, answers the question "how many unit vectors do we need to go
# until we exit the unit cube in this axis."
# i.e. at (0,0) with unit vector (0.7, 0.7), the result should be (1.4, 1.4)
# since after 1.4 unit vectors we're at the exit of the world cube.  
def get_far_val(pts_o: torch.Tensor, pts_d: torch.Tensor, no_nan: bool = False):

    #TODO: This is a hack
    if no_nan:
        pts_d = pts_d + 1e-15

    # Intersection with z = -1, z = 1
    t_z1 = (-1 - pts_o[:, [2]]) / pts_d[:, [2]]
    t_z2 = (1 - pts_o[:, [2]]) / pts_d[:, [2]]
    # Intersection with y = -1, y = 1
    t_y1 = (-1 - pts_o[:, [1]]) / pts_d[:, [1]]
    t_y2 = (1 - pts_o[:, [1]]) / pts_d[:, [1]]
    # Intersection with x = -1, x = 1
    t_x1 = (-1 - pts_o[:, [0]]) / pts_d[:, [0]]
    t_x2 = (1 - pts_o[:, [0]]) / pts_d[:, [0]]

    clipped_ts = torch.cat([torch.maximum(t_z1.clamp(min=0), t_z2.clamp(min=0)),
                            torch.maximum(t_y1.clamp(min=0),
                                          t_y2.clamp(min=0)),
                            torch.maximum(t_x1.clamp(min=0), t_x2.clamp(min=0))], dim=1)
    far_clip = clipped_ts.min(dim=1)[0].unsqueeze(1)
    return far_clip

def get_ray_directions(H, W, newK, dist=None, K=None, sppd=1, with_indices=False):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    When images are already undistorted and rectified, only use newK.
    When images are unrectified, we compute the pixel locations in the undistorted image plane
    and use that to compute ray directions as though a camera with newK intrinsic matrix was used
    to capture the scene. 
    Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a

    Inputs:
        H, W: image height and width
        K: intrinsic matrix of the camera
        dist: distortion coefficients
        newK: output of getOptimalNewCameraMatrix. Compute with free scaling parameter set to 1. 
        This retains all pixels in the undistorted image.         
        sppd: samples per pixel along each dimension. Returns sppd**2 number of rays per pixel
        with_indices: if True, additionally return the i and j meshgrid indices
    Outputs:
        directions: (H*W, 3), the direction of the rays in camera coordinate
        i (optionally) : (H*W, 1) integer pixel coordinate
        j (optionally) : (H*W, 1) integer pixel coordinate
    """
    # offset added so that rays emanate from around center of pixel
    xs = torch.linspace(0, W - 1. / sppd, sppd * W, device=newK.device)
    ys = torch.linspace(0, H - 1. / sppd, sppd * H, device=newK.device)

    grid_x, grid_y = torch.meshgrid([xs, ys])
    grid_x = grid_x.permute(1, 0).reshape(-1, 1)    # (H*W, 1)
    grid_y = grid_y.permute(1, 0).reshape(-1, 1)    # (H*W, 1)

    if dist is not None:
        assert K is not None

        # computing the undistorted pixel locations
        print(f"Computing the undistorted pixel locations to compute the correct ray directions")

        if len(K.shape) == 2:
            K = K.unsqueeze(0)  # (1, 3, 3)

        if isinstance(dist, list):
            dist = torch.tensor(dist, device=K.device).unsqueeze(0)  # (1, 5)

        assert newK is not None, f"Compute newK using getOptimalNewCameraMatrix with alpha=1"
        if len(newK.shape) == 2:
            newK = newK.unsqueeze(0)  # (1, 3, 3)

        points = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)
        undistorted_points = undistort_points(points, K, dist, newK)
        new_grid_x = undistorted_points[0, :, 0:1]
        new_grid_y = undistorted_points[0, :, 1:2]
        newK = newK[0]
    else:
        new_grid_x = grid_x
        new_grid_y = grid_y

    directions = torch.cat([(new_grid_x-newK[0, 2])/newK[0, 0], -(
        new_grid_y-newK[1, 2])/newK[1, 1], -torch.ones_like(grid_x)], -1)  # (H*W, 3)

    if with_indices:
        # returns directions computed from undistorted pixel locations along with the pixel locations in the original distorted image
        return directions, grid_x, grid_y
    else:
        return directions


class LidarRayDirections:
    def __init__(self, lidar_scan: LidarScan, device = 'cpu', chunk_size=512):
        self.lidar_scan = lidar_scan
        self._chunk_size = chunk_size
        self.num_chunks = int(np.ceil(self.lidar_scan.ray_directions.shape[1] / self._chunk_size))
        
    def __len__(self):
        return self.lidar_scan.ray_directions.shape[1]

    def build_lidar_rays(self,
                        lidar_indices: torch.Tensor,
                        ray_range: torch.Tensor,
                        world_cube: WorldCube,
                        lidar_poses: torch.Tensor, # 4x4
                        ignore_world_cube: bool = False) -> torch.Tensor:

        rotate_lidar_opengl = torch.eye(4) #.to(self._device)
        rotate_lidar_points_opengl = torch.eye(3) #.to(self._device)
        if len(lidar_indices)>1:
            depths = self.lidar_scan.distances[lidar_indices] / world_cube.scale_factor
            directions = self.lidar_scan.ray_directions[:, lidar_indices]
            timestamps = self.lidar_scan.timestamps[lidar_indices]
        else:
            depths = self.lidar_scan.distances / world_cube.scale_factor
            directions = self.lidar_scan.ray_directions[:, lidar_indices]
            timestamps = self.lidar_scan.timestamps

        # Now that we're in OpenGL frame, we can apply world cube transformation
        ray_origins: torch.Tensor = lidar_poses[..., :3, 3]
        ray_origins = ray_origins + world_cube.shift
        ray_origins = ray_origins / world_cube.scale_factor
        ray_origins = ray_origins @ rotate_lidar_opengl[:3,:3]
        ray_origins = ray_origins.tile(len(timestamps), 1)

        # print('world_cube shift: ', world_cube.shift, ' scale_factor: ', world_cube.scale_factor)

        # N x 3 x 3 (N homogenous transformation matrices)
        lidar_rotations = lidar_poses[..., :3, :3]
        
        # N x 3 x 1. This takes a 3xN matrix and makes it 1x3xN, then Nx3x1
        directions_3d = directions.unsqueeze(0).swapaxes(0, 2)

        # rotate ray directions from sensor coordinates to world coordinates
        ray_directions = lidar_rotations @ directions_3d

        # ray_directions is now Nx3x1, we want Nx3.
        ray_directions = ray_directions.squeeze()
        # Only now we swap it to opengl coordinates
        ray_directions = ray_directions @ rotate_lidar_points_opengl.T

        if ray_directions.dim()<2:
            ray_directions = torch.unsqueeze(ray_directions, 0)

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

    def fetch_chunk_rays(self, chunk_idx: int, pose: Pose, world_cube: WorldCube, ray_range, ignore_world_cube=False):
        start_idx = chunk_idx*self._chunk_size
        end_idx = min(self.lidar_scan.ray_directions.shape[1], (chunk_idx+1)*self._chunk_size)
        indices = torch.arange(start_idx, end_idx, 1)
        pose_mat = pose.get_transformation_matrix()
        return self.build_lidar_rays(indices, ray_range, world_cube, torch.unsqueeze(pose_mat, 0), ignore_world_cube)[0]

    # Sample distribution is 1D, in row-major order (size of grid_dimensions)
    def sample_chunks(self, sample_distribution: torch.Tensor = None, total_grid_samples = None) -> torch.Tensor:
        
        if total_grid_samples is None:
            total_grid_samples = self.total_grid_samples

        # TODO: This method potentially ignores the upper-right border of each cell. fixme.
        num_grid_cells = self.grid_dimensions[0]*self.grid_dimensions[1]
        
        if sample_distribution is None:
            local_xs = torch.randint(0, self.grid_cell_width, (total_grid_samples,))
            local_ys = torch.randint(0, self.grid_cell_height, (total_grid_samples,))

            local_xs = local_xs.reshape(num_grid_cells, self.samples_per_grid_cell, 1)
            local_ys = local_ys.reshape(num_grid_cells, self.samples_per_grid_cell, 1)

            # num_grid_cells x samples_per_grid_cell x 2
            local_samples = torch.cat((local_ys, local_xs), dim=2)

            # Row-major order
            samples = local_samples + self.cell_offsets

            indices = samples[:,:,0]*self.im_width + samples[:,:,1]
        else:
            local_xs = torch.randint(0, self.grid_cell_width, (total_grid_samples,))
            local_ys = torch.randint(0, self.grid_cell_height, (total_grid_samples,))     
            all_samples = torch.vstack((local_ys, local_xs)).T

            # TODO: There must be a better way
            samples_per_cell: torch.Tensor = sample_distribution * total_grid_samples
            samples_per_cell = samples_per_cell.floor().to(torch.int32)
            remainder = total_grid_samples - samples_per_cell.sum()

            while remainder > len(samples_per_cell):
                breakpoint()
                samples_per_cell += 1
                remainder -= len(samples_per_cell)

            _, best_indices = samples_per_cell.topk(remainder)
            samples_per_cell[best_indices] += 1
            

            repeated_cell_offsets = self.cell_offsets.squeeze(1).repeat_interleave(samples_per_cell, dim=0)
            all_samples += repeated_cell_offsets
            
            indices = all_samples[:,0]*self.im_width + all_samples[:,1]

        return indices


def rays_to_o3d(rays, depths, intensities=None):
    origins = rays[:, :3]
    directions = rays[:, 3:6]
    
    depths = depths.reshape((depths.shape[0], 1))
    end_points = origins + directions*depths

    end_points = end_points.detach().cpu().numpy()

    pcd = o3d.cuda.pybind.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(end_points)

    if intensities is not None:
        intensities = intensities.detach().cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(intensities)

    return pcd

def rays_to_pcd(rays, depths, rays_fname, origins_fname, intensities=None):

    if intensities is None:
        intensities = torch.ones_like(rays[:, :3])
    

    intensity_floats = []
    for intensity_row in intensities:
        red = int(intensity_row[0] * 255).to_bytes(1, 'big', signed=False)
        green = int(intensity_row[1] * 255).to_bytes(1, 'big', signed=False)
        blue = int(intensity_row[2] * 255).to_bytes(1, 'big', signed=False)

        intensity_bytes = struct.pack("4c", red, green, blue, b"\x00")
        intensity_floats.append(struct.unpack('f', intensity_bytes)[0])


    origins = rays[:, :3]
    directions = rays[:, 3:6]
    
    depths = depths.reshape((depths.shape[0], 1))
    end_points = origins + directions*depths
        
    with open(rays_fname, 'w') as f:
        if end_points.shape[0] <= 3:
            end_points = end_points.T
            assert end_points.shape[0] > 3, f"Too few points or wrong shape of pcd file."
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {end_points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {end_points.shape[0]}\n")
        f.write("DATA ascii\n")
        for pt, intensity in zip(end_points, intensity_floats):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {intensity}\n")

    with open(origins_fname, 'w') as f:
        if origins.shape[0] <= 3:
            origins = origins.T
            assert origins.shape[0] > 3, f"Too few points or wrong shape of pcd file."
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4 \n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {origins.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {origins.shape[0]}\n")
        f.write("DATA ascii\n")
        for pt in origins:
            f.write(f"{pt[0]} {pt[1]} {pt[2]} \n")
