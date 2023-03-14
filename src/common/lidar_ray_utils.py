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
        print('rays number: ', self.lidar_scan.ray_directions.shape[1])
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
        depths = self.lidar_scan.distances[lidar_indices] / world_cube.scale_factor
        directions = self.lidar_scan.ray_directions[:, lidar_indices]
        timestamps = self.lidar_scan.timestamps[lidar_indices]

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

    def fetch_chunk_rays(self, chunk_idx: int, pose: Pose, world_cube: WorldCube, ray_range):
        start_idx = chunk_idx*self._chunk_size
        end_idx = min(self.lidar_scan.ray_directions.shape[1], (chunk_idx+1)*self._chunk_size)
        indices = torch.arange(start_idx, end_idx, 1)
        pose_mat = pose.get_transformation_matrix()
        return self.build_lidar_rays(indices, ray_range, world_cube, torch.unsqueeze(pose_mat, 0))[0]

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



# rosbag_path = '/hostroot/home/pckung/fusion_portable/20220216_canteen_day/20220216_canteen_day_ref.bag'
# lidar_topic = '/os_cloud_node/points'

class Mesher(object):
    def __init__(self, model, ckpt, world_cube, rosbag_path, lidar_topic, points_batch_size=500000, ray_batch_size=100000):

        marching_cubes_bound = np.array([[-0,10], [-5,5], [-1,10]])
        # marching_cubes_bound = np.array([[20,20], [20,20], [20,20]])
        self.world_cube_shift = np.expand_dims(world_cube.shift.cpu().numpy(),1)
        self.world_cube_scale_factor = world_cube.scale_factor.cpu().numpy()
        self.marching_cubes_bound = torch.from_numpy((np.array(marching_cubes_bound) + self.world_cube_shift) / self.world_cube_scale_factor)
        self.world_cube = world_cube
        self.model = model
        self.ckpt = ckpt
        self.resolution = 128 #512 #256
        self.clean_mesh_bound_scale = 1.05
        self.points_batch_size = points_batch_size # points_batch_size=500000
        self.ray_batch_size = ray_batch_size
        self.level_set = 10

        self.bag = rosbag.Bag(rosbag_path, 'r')
        self.lidar_topic = lidar_topic
        self.lidar_ts_to_seq_ = self.lidar_ts_to_seq(self.bag, lidar_topic)

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
        bound = self.marching_cubes_bound

        padding = 0.05
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution) # (256,)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        xx, yy, zz = np.meshgrid(x, y, z) # xx: (256, 256, 256)

        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
            dtype=torch.float)
        return {"grid_points": grid_points, "xyz": [x, y, z]}
    
    def get_bound_from_frames_new(self):
        poses = self.ckpt["poses"]
        o3d_pcd = o3d.geometry.PointCloud()
        mesh = o3d.geometry.TriangleMesh()
        tri_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles))
        mesh_frame = o3d.geometry.TriangleMesh()
        py_mesh = pymesh.form_mesh(np.array(mesh.vertices), np.array(mesh.triangles))
        for i, kf in enumerate(poses):
            pose_key = "lidar_pose" # "gt_lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            kf_timestamp = kf["timestamp"].numpy()
            print(kf_timestamp)
            seq = np.argmin(np.abs(np.array(self.lidar_ts_to_seq_) - kf_timestamp))
            lidar_msg = self.find_corresponding_lidar_scan(self.bag, self.lidar_topic, seq)
            lidar_o3d = self.o3d_pc_from_msg(lidar_msg)
            #o3d_pcd = self.merge_o3d_pc(o3d_pcd, lidar_o3d.transform(lidar_pose))
            import copy
            o3d_pcd = copy.deepcopy(lidar_o3d).transform(lidar_pose)

            lidar_position = lidar_pose[:3,3]
            mesh_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position)
            np_pc = np.asarray(lidar_o3d.points)
            theta = np.arctan2(np_pc[:,1], np_pc[:,0])

            slice_num = 1
            theta = theta + np.pi
            angle_seg = 2*np.pi/slice_num
            # mesh = o3d.geometry.TriangleMesh()
            for i in range(slice_num):
                print('i slice: ', i)
                mask = (theta>=(i*angle_seg-0.2)) * (theta <= ((i+1)*angle_seg+0.2))
                np_pc_seg = np_pc[mask]
                np_pc_seg = np.concatenate((np_pc_seg, np.zeros((1,3))), axis=0)
                pcd_seg = o3d.geometry.PointCloud()
                pcd_seg.points = o3d.utility.Vector3dVector(np_pc_seg)
                mesh_seg, _ = pcd_seg.compute_convex_hull()
                # mesh_seg.compute_vertex_normals()
                mesh += mesh_seg.transform(lidar_pose)

                # tri_mesh_seg = trimesh.Trimesh(vertices=np.array(mesh_seg.vertices), faces=np.array(mesh_seg.triangles))
                # # tri_mesh = trimesh.util.concatenate([tri_mesh, tri_mesh_seg])
                # # tri_mesh.remove_duplicate_faces()
                # tri_mesh = trimesh.boolean.union([tri_mesh, tri_mesh_seg])

                py_mesh_seg = pymesh.form_mesh(np.array(mesh_seg.vertices), np.array(mesh_seg.triangles))
                py_mesh = pymesh.boolean(py_mesh, py_mesh_seg, "union")


            #     o3d.visualization.draw_geometries([mesh_frame, pcd_seg, mesh_seg])
            # o3d.visualization.draw_geometries([mesh_frame, o3d_pcd, mesh])
            # o3d.visualization.draw_geometries([mesh_frame, o3d_pcd, tri_mesh.as_open3d.compute_vertex_normals()])
        
        # mesh = tri_mesh.as_open3d

        mesh.vertices = o3d.utility.Vector3dVector(np.array(py_mesh.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(py_mesh.faces))

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)
        
        # o3d.visualization.draw_geometries([o3d_pcd, mesh.compute_vertex_normals()])

        # convert to world cube scale
        mesh = mesh.translate(np.squeeze(self.world_cube_shift))
        mesh.scale(1./self.world_cube_scale_factor, center=(0, 0, 0))

        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)

        return return_mesh

    def mask_with_fov(self, pnts, lidar_positions_list):
        mask_union = np.zeros((pnts.shape[0]))
        for lidar_position in lidar_positions_list:
            out = ((pnts[:,0]-lidar_position[0])**2)/(2.414)**2 + \
                    ((pnts[:,1]-lidar_position[1])**2)/(2.414)**2 - \
                    ((pnts[:,2]-lidar_position[2])**2)
            mask = out>0
            mask_union = np.logical_or(mask_union, mask)
        return mask_union

    def get_bound_from_frames(self):
        poses = self.ckpt["poses"]
        o3d_pcd = o3d.geometry.PointCloud()
        lidar_positions_list = []
        for i, kf in enumerate(poses):
            pose_key = "lidar_pose" # "gt_lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            # print('lidar_pose: \n', lidar_pose)
            kf_timestamp = kf["timestamp"].numpy()
            print(kf_timestamp)
            seq = np.argmin(np.abs(np.array(self.lidar_ts_to_seq_) - kf_timestamp))
            lidar_msg = self.find_corresponding_lidar_scan(self.bag, self.lidar_topic, seq)
            lidar_o3d = self.o3d_pc_from_msg(lidar_msg)
            o3d_pcd = self.merge_o3d_pc(o3d_pcd, lidar_o3d.transform(lidar_pose))

            lidar_position = lidar_pose[:3,3]
            # print('lidar_position: ', lidar_position)
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
        # o3d.visualization.draw_geometries([o3d_pcd, mesh])

        return return_mesh, lidar_positions_list, mesh_lidar_frames

    def get_lidar_position_from_frames(self):
        poses = self.ckpt["poses"]
        lidar_positions_list = []
        for i, kf in enumerate(poses):
            pose_key = "lidar_pose" # "gt_lidar_pose"
            lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
            lidar_position = lidar_pose[:3,3]
            # print('lidar_position: ', lidar_position)
            lidar_position += np.squeeze(self.world_cube_shift)
            lidar_position /= self.world_cube_scale_factor
            lidar_positions_list.append(lidar_position)
            if i!=0:
                mesh_lidar_frames += o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor)
            else:
                mesh_lidar_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=lidar_position*self.world_cube_scale_factor)

        return lidar_positions_list, mesh_lidar_frames

    def eval_points(self, device, xyz_, dir_=None):
        out = self.model.inference(xyz_, dir_)
        return out

    def get_mesh(self, device):
        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            points = points.to(device)
            print("points.shape: ", points.shape)

            lidar_positions_list, mesh_lidar_frames = self.get_lidar_position_from_frames()
            # mesh_bound, lidar_positions_list, mesh_lidar_frames = self.get_bound_from_frames()

            # convex_hull_mask = []
            # print('masking with Convex hull...')
            # for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
            #     convex_hull_mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            # convex_hull_mask = np.concatenate(convex_hull_mask, axis=0)
            
            lidar_fov_mask = []
            print('masking with Lidar FOV...')
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                lidar_fov_mask.append(self.mask_with_fov(pnts.cpu().numpy(), lidar_positions_list))
            lidar_fov_mask = np.concatenate(lidar_fov_mask, axis=0)

            z = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                print('pnts.shape: ', pnts.shape)
                z.append(self.eval_points(device, pnts).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)
            print("z.shape: ", z.shape) # (16777216,) = 256^3

            # z[z>-5000]=1000
            # np.set_printoptions(threshold=sys.maxsize)
            # print("z: ", z)
            # z[~convex_hull_mask] = -1000
            z[~lidar_fov_mask] = -1000
            z = z.astype(np.float32)
            z[z<0]=-1000

            volume = np.copy(z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]))
            # print("volume: ", volume)
            print('volume.shape: ', volume.shape)
            # np.save('./volume.npy', volume)

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
            
            vertex_colors = None
            vertices *= self.world_cube_scale_factor
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh_o3d = mesh.as_open3d
            mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=1)
            mesh_o3d.compute_vertex_normals()
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.squeeze(self.world_cube_shift))
            o3d.visualization.draw_geometries([mesh_o3d, mesh_lidar_frames],
                                                mesh_show_back_face=True, mesh_show_wireframe=False,
                                                # lookat=np.array([-0.048256088124968066, -5.429726350336284, 0.50703743111366784 ]),
                                                # up=np.array([ -0.16608120692279579, 0.088253514094586788, 0.98215495211143966 ]),
                                                # front=np.array([ -0.96666810505652834, 0.18223917196167946, -0.17983786828458104 ]),
                                                # zoom=np.array(0.21999999999999958)
                                                lookat=np.array([ 0.69733560997223265, -5.0347382973578814, 2.3461460184924174 ]),
                                                up=np.array([ 0.5014307626075396, 0.16918213207473837, 0.84849548996884305 ]),
                                                front=np.array([ -0.75505795871470938, -0.39326278779528995, 0.52462544612042328 ]),
                                                zoom=np.array(0.21999999999999958)
                                                )


            # {
            # "class_name" : "ViewTrajectory",
            # "interval" : 29,
            # "is_loop" : false,
            # "trajectory" : 
            # [
            #     {
            #         "boundingbox_max" : [ 9.3998233003220566, 4.0183541413433321, 13.900876649258752 ],
            #         "boundingbox_min" : [ -29.031641540527342, -26.822685775756835, -5.995273065211542 ],
            #         "field_of_view" : 60.0,
            #         "front" : [ -0.75505795871470938, -0.39326278779528995, 0.52462544612042328 ],
            #         "lookat" : [ 0.69733560997223265, -5.0347382973578814, 2.3461460184924174 ],
            #         "up" : [ 0.5014307626075396, 0.16918213207473837, 0.84849548996884305 ],
            #         "zoom" : 0.21999999999999958
            #     }
            # ],
            # "version_major" : 1,
            # "version_minor" : 0
            # }
            # {
            #     "class_name" : "ViewTrajectory",
            #     "interval" : 29,
            #     "is_loop" : false,
            #     "trajectory" : 
            #     [
            #         {
            #             "boundingbox_max" : [ 9.3998193272356492, 4.0183535737595593, 13.900793579973287 ],
            #             "boundingbox_min" : [ -29.031641540527342, -26.822685775756835, -5.9952813631679343 ],
            #             "field_of_view" : 60.0,
            #             "front" : [ -0.39272999367830352, -0.33181713635021692, 0.85770655826441089 ],
            #             "lookat" : [ 0.69733560997223265, -5.0347382973578814, 2.3461460184924174 ],
            #             "up" : [ 0.77526001287952107, 0.38224156053495595, 0.50285514994861757 ],
            #             "zoom" : 0.11999999999999962
            #         }
            #     ],
            #     "version_major" : 1,
            #     "version_minor" : 0
            # }


            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_o3d.vertices))
            # pcd.estimate_normals()
            # o3d.visualization.draw_geometries([pcd])

            # mesh_out_file = ''
            # mesh.export(mesh_out_file)
            # print('Saved mesh at', mesh_out_file)