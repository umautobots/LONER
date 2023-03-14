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

class Mesher(object):
    def __init__(self, model, ckpt, world_cube, rosbag_path=None, lidar_topic=None,
                       resolution = 0.2, marching_cubes_bound = [[-40,20], [0,20], [-3,15]], level_set=10,
                       points_batch_size=500000):

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
        print("get_grid_uniform")

        bound = torch.from_numpy((np.array(self.marching_cubes_bound) + np.expand_dims(self.world_cube_shift,1)) / self.world_cube_scale_factor)

        length = self.marching_cubes_bound[:,1]-self.marching_cubes_bound[:,0]
        num = (length/resolution).astype(int)

        print(num)
        x = np.linspace(bound[0][0], bound[0][1],num[0])
        y = np.linspace(bound[1][0], bound[1][1],num[1])
        z = np.linspace(bound[2][0], bound[2][1],num[2])

        xx, yy, zz = np.meshgrid(x, y, z) # xx: (256, 256, 256)

        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
            dtype=torch.float)
        return {"grid_points": grid_points, "xyz": [x, y, z]}
    
    def mask_with_fov(self, pnts, lidar_positions_list):
        mask_union = np.zeros((pnts.shape[0]))
        for lidar_position in lidar_positions_list:
            out = ((pnts[:,0]-lidar_position[0])**2)/(2.414)**2 + \
                    ((pnts[:,1]-lidar_position[1])**2)/(2.414)**2 - \
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

    def get_mesh(self, device, sigma_only=True, threshold=0, use_lidar_fov_mask=True, use_convex_hull_mask=False):
        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            points = points.to(device)
            print("points.shape: ", points.shape)
            
            # inference points
            print('inferring grid points...')
            z = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(self.model.inference_points(pnts, dir_=None, sigma_only=True).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)

            if use_lidar_fov_mask:
                lidar_fov_mask = []
                print('Masking with Lidar FOV...')
                lidar_positions_list, mesh_lidar_frames = self.get_lidar_positions(use_gt_poses=False)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    lidar_fov_mask.append(self.mask_with_fov(pnts.cpu().numpy(), lidar_positions_list))
                lidar_fov_mask = np.concatenate(lidar_fov_mask, axis=0)
                z[~lidar_fov_mask] = -1000
            
            if use_convex_hull_mask:
                self.bag = rosbag.Bag(self.rosbag_path, 'r')
                self.lidar_ts_to_seq_ = self.lidar_ts_to_seq(self.bag, self.lidar_topic)
                mesh_bound, lidar_positions_list, mesh_lidar_frames = self.get_bound_from_lidar_convex_hull(use_gt_poses=False)
                convex_hull_mask = []
                print('Masking with Convex hull...')
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    convex_hull_mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                convex_hull_mask = np.concatenate(convex_hull_mask, axis=0)
                z[~convex_hull_mask] = -1000

            z = z.astype(np.float32)
            z[z<threshold]=-1000
            volume = np.copy(z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]))
            print('volume.shape: ', volume.shape)

            # marching cube
            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
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
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                color = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    dir_ = pnts # always view from orgin
                    color.append(self.model.inference_points(pnts, dir_=dir_, sigma_only=False).cpu().numpy()[:, :3])
                vertex_colors = np.concatenate(color, axis=0)

            vertices *= self.world_cube_scale_factor
            vertices -= self.world_cube_shift

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            if not sigma_only:
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.squeeze(self.world_cube_shift))
            return mesh_o3d, mesh_lidar_frames, origin_frame