import struct

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
from common.sensors import Image
from common.settings import Settings
from common.pose_utils import WorldCube
from common.sensors import Image, LidarScan
from common.ray_utils import LidarRayDirections

from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
import rosbag
import rospy
import trimesh
import pymesh
from packaging import version
import skimage
from models.model_tcnn import Model, OccupancyGridModel


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

class Mesher(object):
    def __init__(self, model, ckpt, world_cube, ray_range, rosbag_path=None, lidar_topic=None,
                       resolution = 0.2, marching_cubes_bound = [[-40,20], [0,20], [-3,15]], level_set=10,
                       points_batch_size=5000000, lidar_vertical_fov = [-22.5, 22.5]):

        # marching_cubes_bound = np.array([[-40,20], [-20,50], [-3,15]])
        self.marching_cubes_bound = np.array(marching_cubes_bound)
        self.world_cube_shift = world_cube.shift.cpu().numpy()
        self.world_cube_scale_factor = world_cube.scale_factor.cpu().numpy()
        self.world_cube = world_cube
        self.model = model
        self.ckpt = ckpt
        self.resolution = resolution
        self.points_batch_size = points_batch_size
        self.level_set = level_set

        self.rosbag_path = rosbag_path
        self.lidar_topic = lidar_topic
        self.clean_mesh_bound_scale = 1.05

        self.ray_range = ray_range

        self.lidar_vertical_fov = lidar_vertical_fov

    def lidar_ts_to_seq(self, bag, lidar_topic):
        init_ts = -1
        lidar_ts_to_seq = []
        for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
            if init_ts == -1:
                init_ts = msg.header.stamp.to_sec() # TBV
            timestamp = msg.header.stamp.to_sec() - init_ts
            lidar_ts_to_seq.append(timestamp)
        return lidar_ts_to_seq

    def find_corresponding_lidar_scan(self, bag, lidar_topic, seq):
        for topic, msg, ts in bag.read_messages(topics=[lidar_topic]):
            if msg.header.seq == seq:
                return msg

    def o3d_pc_from_msg(self, lidar_msg: PointCloud2):
        lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
        lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
        xyz = lidar_data[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd

    def merge_o3d_pc(self, pcd1, pcd2):
        pcd = o3d.geometry.PointCloud()
        p1_load = np.asarray(pcd1.points)
        p2_load = np.asarray(pcd2.points)
        p3_load = np.concatenate((p1_load, p2_load), axis=0)
        pcd.points = o3d.utility.Vector3dVector(p3_load)
        p1_color = np.asarray(pcd1.colors)
        p2_color = np.asarray(pcd2.colors)
        p3_color = np.concatenate((p1_color, p2_color), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(p3_color)
        return pcd


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
    
    def mask_with_fov(self, pnts, lidar_positions_list, vertical_angle=22.5):
        mask_union = np.zeros((pnts.shape[0]))
        for lidar_position in lidar_positions_list:
            alpha = 1/np.tan(vertical_angle/180*np.pi)
            # print('vertical_angle ', vertical_angle, 'alpha ', alpha)
            out = ((pnts[:,0]-lidar_position[0])**2)/(alpha)**2 + \
                    ((pnts[:,1]-lidar_position[1])**2)/(alpha)**2 - \
                    ((pnts[:,2]-lidar_position[2])**2)
            mask = out>0
            mask_union = np.logical_or(mask_union, mask)
        return mask_union

    def get_bound_from_lidar_convex_hull(self, use_gt_poses=False):
        poses = self.ckpt["poses"]
        o3d_pcd = o3d.geometry.PointCloud()
        lidar_positions_list = []
        for i, kf in enumerate(poses):
            if use_gt_poses:
                pose_key = "gt_lidar_pose"
            else:
                pose_key = "lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            kf_timestamp = kf["timestamp"].numpy()
            print(kf_timestamp)
            seq = np.argmin(np.abs(np.array(self.lidar_ts_to_seq_) - kf_timestamp))
            lidar_msg = self.find_corresponding_lidar_scan(self.bag, self.lidar_topic, seq)
            lidar_o3d = self.o3d_pc_from_msg(lidar_msg)
            o3d_pcd = self.merge_o3d_pc(o3d_pcd, lidar_o3d.transform(lidar_pose))

            lidar_position = lidar_pose[:3,3]
            lidar_position += np.squeeze(self.world_cube_shift)
            lidar_position /= self.world_cube_scale_factor
            lidar_positions_list.append(lidar_position)

            if i!=0:
                mesh_lidar_frames += o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor)
            else:
                mesh_lidar_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor)
            # o3d.visualization.draw_geometries([mesh_frame, o3d_pcd])

        mesh, _ = o3d_pcd.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)

        mesh = mesh.translate(np.squeeze(self.world_cube_shift))
        mesh.scale(1./self.world_cube_scale_factor, center=(0, 0, 0))

        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        o3d.visualization.draw_geometries([o3d_pcd, mesh])

        return return_mesh, lidar_positions_list, mesh_lidar_frames
    
    def get_lidar_map(self, use_gt_poses=False):
        poses = self.ckpt["poses"]
        o3d_pcd = o3d.geometry.PointCloud()
        lidar_map_list = []
        for i, kf in enumerate(poses):
            if use_gt_poses:
                pose_key = "gt_lidar_pose"
            else:
                pose_key = "lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            kf_timestamp = kf["timestamp"].numpy()
            print(kf_timestamp)
            seq = np.argmin(np.abs(np.array(self.lidar_ts_to_seq_) - kf_timestamp))
            lidar_msg = self.find_corresponding_lidar_scan(self.bag, self.lidar_topic, seq)
            # lidar_o3d = self.o3d_pc_from_msg(lidar_msg)
            # o3d_pcd = self.merge_o3d_pc(o3d_pcd, lidar_o3d.transform(lidar_pose))
            lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
            lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
            xyz = lidar_data[:, :3]
            xyz_h = np.concatenate((xyz, np.ones((xyz.shape[0],1))), axis=1)
            xyz = (xyz_h @ lidar_pose.T)[:,:-1]
            lidar_map_list.append(xyz)
        lidar_map = np.concatenate(lidar_map_list, axis=0)
        print('lidar_map.shape: ', lidar_map.shape)
        o3d_pcd.points = o3d.utility.Vector3dVector(lidar_map)
        o3d_pcd = o3d_pcd.voxel_down_sample(self.resolution*1)
        return o3d_pcd
    
    def mask_with_pc(self, pnts, lidar_map_kd_tree):
        mask = np.zeros((pnts.shape[0]))
        distances, _ = lidar_map_kd_tree.query(pnts)
        mask = distances < 0.1 # (m)
        return mask

    def get_lidar_positions(self, use_gt_poses=False):
        poses = self.ckpt["poses"]
        lidar_positions_list = []
        for i, kf in enumerate(poses):
            if use_gt_poses:
                pose_key = "gt_lidar_pose"
            else:
                pose_key = "lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            lidar_position = lidar_pose[:3,3]
            lidar_position += np.squeeze(self.world_cube_shift)
            lidar_position /= self.world_cube_scale_factor
            lidar_positions_list.append(lidar_position)
            if i!=0:
                mesh_lidar_frames += o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor-self.world_cube_shift)
            else:
                mesh_lidar_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor-self.world_cube_shift)

        return lidar_positions_list, mesh_lidar_frames

    def eval_points(self, device, xyz_, dir_=None):
        out = self.model.inference_points(xyz_, dir_)
        return out

    def get_mesh(self, device, ray_sampler, occupancy_grid, occ_voxel_size, sigma_only=True, threshold=0, 
                 use_lidar_fov_mask=False, use_convex_hull_mask=False, use_lidar_pointcloud_mask=False, use_occ_mask=False, \
                    color_mesh_extraction_method='direct_point_query', use_weights = False, skip_step = 15,
                    var_threshold=None):

        if use_weights:
            mask_val = 0
        else:
            mask_val = -1000
        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            points = points.to(device)
            print("points.shape: ", points.shape)
            
            if use_weights:
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

                grid_pts = grid["grid_points"]

                results = torch.zeros((len(points),), dtype=float, device=device)
                print(points.shape)

                for pose_state in tqdm(lidar_poses):
                    pose_key = "lidar_pose"
                    lidar_pose = Pose(pose_tensor=pose_state[pose_key]).to(device)

                    size = ray_directions.lidar_scan.ray_directions.shape[1]

                    samples = torch.zeros((0,2), device=device, dtype=torch.float32)
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
                        
                        # good_weights = torch.logical_and(weights.flatten() > threshold, bucket_idx < len(results))
                        # good_weights = torch.logical_and(bucket_idx < len(results))
                        valid_buckets = bucket_idx < len(results) # Hack around bucketize edge cases
                        weights = weights[valid_buckets]
                        bucket_idx = bucket_idx[valid_buckets]
                        
                        results[bucket_idx] = torch.max(results[bucket_idx], weights.flatten())
                        
                results = results.cpu().numpy()
            else:
                # inference points
                print('inferring grid points...')
                results = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    results.append(self.model.inference_points(pnts, dir_=None, sigma_only=True).cpu().numpy()[:, -1])
                results = np.concatenate(results, axis=0)
                # results = np.ones_like(results) * 1000

            mesh_lidar_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0,0,0))
            if use_lidar_fov_mask:
                lidar_fov_mask = []
                print('Masking with Lidar FOV...')
                lidar_positions_list, mesh_lidar_frames = self.get_lidar_positions(use_gt_poses=False)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    lidar_fov_mask.append(self.mask_with_fov(pnts.cpu().numpy(), lidar_positions_list))
                lidar_fov_mask = np.concatenate(lidar_fov_mask, axis=0)
                results[~lidar_fov_mask] = mask_val
            
            if use_convex_hull_mask:
                self.bag = rosbag.Bag(self.rosbag_path, 'r')
                self.lidar_ts_to_seq_ = self.lidar_ts_to_seq(self.bag, self.lidar_topic)
                mesh_bound, lidar_positions_list, mesh_lidar_frames = self.get_bound_from_lidar_convex_hull(use_gt_poses=False)
                convex_hull_mask = []
                print('Masking with Convex hull...')
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    convex_hull_mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                convex_hull_mask = np.concatenate(convex_hull_mask, axis=0)
                results[~convex_hull_mask] = mask_val

            lidar_map = o3d.geometry.PointCloud()
            if use_lidar_pointcloud_mask:
                print('Masking with Lidar map...')
                self.bag = rosbag.Bag(self.rosbag_path, 'r')
                self.lidar_ts_to_seq_ = self.lidar_ts_to_seq(self.bag, self.lidar_topic)
                lidar_map = self.get_lidar_map(use_gt_poses=False)
                
                lidar_map_mask = []
                from scipy.spatial import cKDTree as KDTree
                lidar_map_np = np.asarray(lidar_map.points)
                lidar_map_kd_tree = KDTree(lidar_map_np)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    lidar_map_mask.append(self.mask_with_pc((pnts.cpu().numpy()*self.world_cube_scale_factor - self.world_cube_shift), lidar_map_kd_tree))
                lidar_map_mask = np.concatenate(lidar_map_mask, axis=0)
                results[~lidar_map_mask] = mask_val

            if use_occ_mask:
                print('Masking with OCC...')
                occ_sigma_np = occupancy_grid.squeeze().cpu().detach().numpy()
                if occ_sigma_np.sum() > 1:
                    occ_probs = 1. / (1 + np.exp(-occ_sigma_np))
                    occ_probs = (510 *  (occ_probs.clip(0.5, 1.0) - 0.5)).astype(np.uint8).reshape(-1)
                    nonzero_indices = occ_probs.nonzero()
                    print('occ_voxel_size', occ_voxel_size)
                    x_ = np.arange(occ_voxel_size)
                    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
                    # X = np.stack([x.reshape(-1)[nonzero_indices], y.reshape(-1)[nonzero_indices], -z.reshape(-1)[nonzero_indices], occ_probs[nonzero_indices]], axis=1)
                    X = np.stack([x.reshape(-1)[nonzero_indices], y.reshape(-1)[nonzero_indices], z.reshape(-1)[nonzero_indices], occ_probs[nonzero_indices]], axis=1)
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(X[:,:3]-(occ_voxel_size/2.))
                    # pcd.colors = o3d.utility.Vector3dVector(np.repeat(X[:,3].reshape((-1,1)), 3, axis=1)/255.)
                    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
                    # o3d.visualization.draw_geometries([pcd, mesh_frame])

                    occ_mask = []
                    for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                        print(pnts.shape)
                        point_logits = OccupancyGridModel.interpolate(ray_sampler._occ_gamma, torch.unsqueeze(pnts,0))
                        point_probs = 1. / (1 + torch.exp(-point_logits))
                        point_probs = 2 * (point_probs.clamp(min=0.5, max=1.0) - 0.5)
                        point_logits = torch.squeeze(point_logits)
                        occ_mask.append((point_logits > 0).cpu().detach().numpy())
                    occ_mask = np.concatenate(occ_mask, axis=0)
                    results[~occ_mask] = mask_val

            results = results.astype(np.float32)
            results[results<threshold]=mask_val
            volume = np.copy(results.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]))
            print('volume.shape: ', volume.shape)

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

            if sigma_only:
                vertex_colors = None
            else:
                points = torch.from_numpy(vertices)
                # color is extracted by passing the coordinates of mesh vertices through the network
                if color_mesh_extraction_method == 'direct_point_query':
                    color = []
                    print('inferring color by point inquery')
                    for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                        # dir_ = torch.from_numpy(dirs[i*self.points_batch_size: (i+1)*self.points_batch_size])
                        dir_ = pnts
                        color.append(self.model.inference_points(pnts, dir_=dir_, sigma_only=False).cpu().numpy()[:, :3])
                    vertex_colors = np.concatenate(color, axis=0)
                elif color_mesh_extraction_method == 'render_ray':
                    print('getting view direction')
                    dirs, dists, corr_lidar_idx_list = self.get_dir(vertices, lidar_positions_list)
                    print('inferring color by render ray')
                    vertex_colors = np.zeros_like(vertices)
                    lidar_scans, used_lidar_positions, vertices_idxs = self.get_lidar_sequence_for_color_rendering(
                                                        vertices, dirs, dists, corr_lidar_idx_list, lidar_positions_list)
                    
                    CHUNK_SIZE=512*3
                    for i, z in enumerate(zip(lidar_scans, used_lidar_positions, vertices_idxs)):
                        lidar_scan, used_lidar_position, vertices_idx = z
                        print('lidar id: ', i)
                        lidar_pose = torch.eye(4).to(device)
                        lidar_pose[:3,3] = torch.from_numpy(used_lidar_position)
                        ray_directions = LidarRayDirections(lidar_scan.to(device), chunk_size=CHUNK_SIZE, device=device)

                        size = lidar_scan.ray_directions.shape[1]
                        rgb_fine = torch.zeros((size,3), dtype=torch.float32).view(-1, 3)
                        depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)
                        for chunk_idx in range(ray_directions.num_chunks):
                            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, Pose(transformation_matrix = lidar_pose).to(device),
                                                                         WorldCube(1, 0), (0, 1), ignore_world_cube=True)
                            eval_rays = eval_rays.to(device)
                            results = self.model(eval_rays, ray_sampler, scale_factor=1, testing=True)

                            rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
                            depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)
                        
                        vertex_colors[vertices_idx] = rgb_fine.cpu().numpy()

            vertices *= self.world_cube_scale_factor
            vertices -= self.world_cube_shift

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            if not sigma_only:
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            return mesh_o3d, mesh_lidar_frames

    def get_lidar_sequence_for_color_rendering(self, vertices, dirs, dists, idx_list, lidar_positions_list):
        lidar_scan_list = []
        used_lidar_positions_list = []
        points_idx_list = []
        idx_list = np.array(idx_list)
        for i in range(np.max(np.array(idx_list))):
            points_mask = np.squeeze(idx_list == i)
            dirs_ = torch.from_numpy(dirs[points_mask].T)
            dists_ = torch.from_numpy(dists[points_mask].T)
            timestamps = torch.zeros(dirs_.shape[1])
            lidar_scan = LidarScan(dirs_.float(), torch.squeeze(dists_).float(), timestamps.float())
            points_idx = np.where(points_mask == True)[0]
            lidar_scan_list.append(lidar_scan)
            used_lidar_positions_list.append(lidar_positions_list[i])
            points_idx_list.append(points_idx)
        return lidar_scan_list, used_lidar_positions_list, points_idx_list

    def get_dir(self, vertices, lidar_positions_list):
        lidar_positions = np.array(lidar_positions_list)
        o3d_lidar_position = o3d.geometry.PointCloud()
        o3d_lidar_position.points = o3d.utility.Vector3dVector(lidar_positions)
        o3d_vertice = o3d.geometry.PointCloud()
        o3d_vertice.points = o3d.utility.Vector3dVector(vertices)

        kd_tree = o3d.geometry.KDTreeFlann(o3d_lidar_position)
        dirs = []
        idx_list = []
        for i in range(vertices.shape[0]):
            [_, idx, _] = kd_tree.search_knn_vector_3d(o3d_vertice.points[i], 1)
            closest_lidar_position = lidar_positions[np.asarray(idx)]
            dir = np.expand_dims(vertices[i],0) - closest_lidar_position
            dirs.append(dir)
            idx_list.append(idx)
        dirs = np.concatenate(dirs, axis=0)
        dists = np.expand_dims(np.sqrt(np.sum(dirs**2,1)), 1)
        dirs /= dists # L2 norm

        return dirs, dists, idx_list