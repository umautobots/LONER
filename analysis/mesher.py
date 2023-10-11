import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from kornia.geometry.calibration import undistort_points

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from common.pose import Pose
from common.pose_utils import WorldCube
from common.sensors import LidarScan
from common.ray_utils import LidarRayDirections

from sensor_msgs.msg import PointCloud2
import pandas as pd
import ros_numpy
import rosbag
import trimesh
from packaging import version
import skimage


def build_lidar_scan(lidar_intrinsics):
    vert_fov = lidar_intrinsics["vertical_fov"]
    vert_res = lidar_intrinsics["vertical_resolution"]
    hor_res = lidar_intrinsics["horizontal_resolution"]

    phi = torch.arange(vert_fov[0], vert_fov[1], vert_res).deg2rad()
    theta = torch.arange(0, 360, hor_res).deg2rad()

    phi_grid, theta_grid = torch.meshgrid(phi, theta)

    phi_grid = torch.pi/2 - phi_grid.reshape(-1, 1)
    theta_grid = theta_grid.reshape(-1, 1)

    x = torch.cos(theta_grid) * torch.sin(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(phi_grid)

    xyz = torch.hstack((x,y,z))

    scan = LidarScan(xyz.T, torch.ones_like(x).flatten(), torch.zeros_like(x).flatten()).to(0)

    return scan 

# Some of the code is borrowed from the nice-slam mesher: https://github.com/cvg/nice-slam/blob/master/src/utils/Mesher.py
class Mesher(object):
    def __init__(self, model, ckpt, world_cube, ray_range,
                       resolution = 0.2, marching_cubes_bound = [[-40,20], [0,20], [-3,15]], level_set=0,
                       points_batch_size=5000000, lidar_vertical_fov = [-22.5, 22.5]):

        self.marching_cubes_bound = np.array(marching_cubes_bound)
        self.world_cube_shift = world_cube.shift.cpu().numpy()
        self.world_cube_scale_factor = world_cube.scale_factor.cpu().numpy()
        self.world_cube = world_cube
        self.model = model
        self.ckpt = ckpt
        self.resolution = resolution
        self.points_batch_size = points_batch_size
        self.level_set = level_set
        self.ray_range = ray_range
        self.lidar_vertical_fov = lidar_vertical_fov

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """

        bound = torch.from_numpy((np.array(self.marching_cubes_bound) + np.expand_dims(self.world_cube_shift,1)) / self.world_cube_scale_factor)

        length = self.marching_cubes_bound[:,1]-self.marching_cubes_bound[:,0]
        num = (length/resolution).astype(int)
        print("Requested Size:", num)

        x = np.linspace(bound[0][0], bound[0][1],num[0])
        y = np.linspace(bound[1][0], bound[1][1],num[1])
        z = np.linspace(bound[2][0], bound[2][1],num[2])

        xx, yy, zz = np.meshgrid(x, y, z) # xx: (256, 256, 256)

        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
            dtype=torch.float)
        return {"grid_points": grid_points, "xyz": [x, y, z]}

    def eval_points(self, device, xyz_, dir_=None):
        out = self.model.inference_points(xyz_, dir_)
        return out

    def get_mesh(self, device, ray_sampler, skip_step = 15, var_threshold=None):

        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            points = points.to(device)
            
            lidar_intrinsics = {
                "vertical_fov": self.lidar_vertical_fov,
                "vertical_resolution": 0.25,
                "horizontal_resolution": 0.25
            }

            scan = build_lidar_scan(lidar_intrinsics)
            ray_directions = LidarRayDirections(scan)

            poses = self.ckpt["poses"]    
            lidar_poses = poses[::skip_step]

            bound = torch.from_numpy((np.array(self.marching_cubes_bound) + np.expand_dims(self.world_cube_shift,1)) / self.world_cube_scale_factor)
            
            x_boundaries = torch.from_numpy(grid["xyz"][0]).contiguous().to(device)
            y_boundaries = torch.from_numpy(grid["xyz"][1]).contiguous().to(device)
            z_boundaries = torch.from_numpy(grid["xyz"][2]).contiguous().to(device)

            results = torch.zeros((len(points),), dtype=float, device=device)

            for pose_state in tqdm(lidar_poses):
                pose_key = "lidar_pose"
                lidar_pose = Pose(pose_tensor=pose_state[pose_key]).to(device)

                for chunk_idx in range(ray_directions.num_chunks):
                    eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, self.world_cube, self.ray_range)
                    eval_rays = eval_rays.to(device)
                    model_result = self.model(eval_rays, ray_sampler, self.world_cube_scale_factor, testing=False, return_variance=True)

                    spoints = model_result["points_fine"].detach()
                    weights = model_result["weights_fine"].detach()
                    variance = model_result["variance"].detach().view(-1,)
                    depths = model_result["depth_fine"].detach().view(-1,)

                    valid_idx = depths < self.ray_range[1] - 0.25

                    if var_threshold is not None:
                        valid_idx = torch.logical_and(valid_idx, variance < var_threshold)

                    spoints = spoints[valid_idx, ...]
                    weights = weights[valid_idx, ...]

                    spoints = spoints.view(-1, 3)
                    weights = weights.view(-1, 1)

                    good_idx = torch.ones_like(weights.flatten())
                    for i in range(3):
                        good_dim = torch.logical_and(spoints[:,i] >= bound[i][0], spoints[:,i] <= bound[i][1])
                        good_idx = torch.logical_and(good_idx, good_dim)

                    spoints = spoints[good_idx]

                    if len(spoints) == 0:
                        continue

                    x = spoints[:,0].contiguous()
                    y = spoints[:,1].contiguous()
                    z = spoints[:,2].contiguous()
                    
                    x_buck = torch.bucketize(x, x_boundaries)
                    y_buck = torch.bucketize(y, y_boundaries)
                    z_buck = torch.bucketize(z, z_boundaries)

                    bucket_idx = x_buck*len(z_boundaries) + y_buck * len(x_boundaries)*len(z_boundaries) + z_buck
                    weights = weights[good_idx]
                    
                    valid_buckets = bucket_idx < len(results) # Hack around bucketize edge cases
                    weights = weights[valid_buckets]
                    bucket_idx = bucket_idx[valid_buckets]
                    
                    results[bucket_idx] = torch.max(results[bucket_idx], weights.flatten())
                        
            results = results.cpu().numpy()
            results = results.astype(np.float32)
            volume = np.copy(results.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]))

            # marching cube
            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=results.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=results.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print(
                    'marching_cubes error. Possibly no surface extracted from the level set.'
                )
                return

            # convert back to world coordinates
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            vertices *= self.world_cube_scale_factor
            vertices -= self.world_cube_shift

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

            return mesh_o3d