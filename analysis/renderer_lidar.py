#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pathlib
import pickle
import re
import sys
import time
import torch
import tqdm
import rosbag
import rospy

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

import open3d as o3d

from render_utils import *

from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.ray_utils import CameraRayDirections
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler
from src.common.pose_utils import create_spiral_poses

# autopep8: on

from src.common.sensors import Image, LidarScan
from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
from src.common.lidar_ray_utils import LidarRayDirections, Mesher, get_far_val

assert torch.cuda.is_available(), 'Unable to find GPU'

CHUNK_SIZE=512

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--eval", default=False, dest="eval", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
parser.add_argument("--only_last_frame", default=False, dest="only_last_frame", action="store_true")

args = parser.parse_args()

checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

if args.ckpt_id is None:
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"

checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")

# render_dir = pathlib.Path(f"{args.experiment_directory}/renders/{checkpoint}_gt_poses_{args.use_gt_poses}")
# os.makedirs(render_dir, exist_ok=True)

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

intrinsic = full_config.calibration.camera_intrinsic
im_size = torch.Tensor([intrinsic.height, intrinsic.width]) #/ 2
# full_config["calibration"]["camera_intrinsic"]["height"] = int(im_size[0])
# full_config["calibration"]["camera_intrinsic"]["width"] = int(im_size[1])

if args.debug:
    full_config['debug'] = True


# TODO (Seth): Manual Seed
# seed = cfg.seed
torch.backends.cudnn.enabled = True
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

torch.backends.cudnn.benchmark = True
# rng = default_rng(seed)
_DEVICE = torch.device(full_config.mapper.device)


torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()

occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"

scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
shift = full_config.world_cube.shift
world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

# ray_directions = LidarRayDirections(chunk_size=CHUNK_SIZE, device=_DEVICE)
# ray_directions = CameraRayDirections(full_config.calibration, chunk_size=CHUNK_SIZE, device=_DEVICE)

# Returns the 3D logits as a 5D tensor
occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)

ray_sampler = OccGridRaySampler()

# use single fine MLP when using OGM
model_config = full_config.mapper.optimizer.model_config.model
model = Model(model_config).to(_DEVICE)


print(f'Loading checkpoint from: {checkpoint_path}')
ckpt = torch.load(str(checkpoint_path))
model.load_state_dict(ckpt['network_state_dict'])

cfg = full_config.mapper.optimizer.model_config
ray_range = cfg.data.ray_range


occ_model.load_state_dict(ckpt['occ_model_state_dict'])
# initialize occ_sigma
occupancy_grid = occ_model()
ray_sampler.update_occ_grid(occupancy_grid.detach())

def render_dataset_frame(pose: Pose):
    tic_img = time.time()
    size = (int(im_size[0]), int(im_size[1]), 1)
    rgb_size = (int(im_size[0]), int(im_size[1]), 3)
    rgb_fine = torch.zeros(rgb_size, dtype=torch.float32).view(-1, 3)
    depth_fine = torch.zeros(size, dtype=torch.float32).view(-1, 1)
    peak_depth_consistency = torch.zeros(size, dtype=torch.float32).view(-1, 1)
    print("--------------------")
    print("render_dataset_frame")

    for chunk_idx in range(ray_directions.num_chunks):
        # tic = time.time()
        eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, pose, world_cube, ray_range)
        eval_rays = eval_rays.to(_DEVICE)

        results = model(eval_rays, ray_sampler, scale_factor, testing=True)

        rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
        depth = results['depth_fine'].unsqueeze(1)
        depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)

        s_vals = results['samples_fine']
        weights_pred = results['weights_fine']
        s_peaks = s_vals[torch.arange(eval_rays.shape[0]), weights_pred.argmax(dim=1)].unsqueeze(1)
        peak_depth_consistency[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = torch.abs(s_peaks - depth)

    rgb_fine = rgb_fine.reshape(1, rgb_size[0] , rgb_size[1], rgb_size[2]).permute(0, 3, 1, 2)
    depth_fine = depth_fine.reshape(1, size[0], size[1] , 1).permute(0, 3, 1, 2) * scale_factor
    peak_depth_consistency = peak_depth_consistency.reshape(1, size[0], size[1], 1).permute(0, 3, 1, 2) * scale_factor
    print(f'Took: {time.time() - tic_img} seconds for rendering an image')
    
    return rgb_fine.clamp(0, 1), depth_fine, peak_depth_consistency

def render_spiral(center_pose: Pose):
    rgbs = []
    depths = []

    focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                    # given in the original repo. Mathematically if near=1
                    # and far=infinity, then this number will converge to 4

    radii = torch.Tensor([1, 1, 1]) * scale_factor
    spiral_poses = create_spiral_poses(radii, focus_depth, n_poses=5, homogenous=True)

    render_poses = (center_pose.get_transformation_matrix() @ spiral_poses)[...,:3,:]

    for pose in tqdm.tqdm(render_poses):
        rgb, depth, _ = render_dataset_frame(Pose(pose))

        rgbs.append(rgb.detach().cpu().numpy().reshape((int(im_size[0]), int(im_size[1]), 3)) * 255)
        depths.append(depth.detach().squeeze().cpu().numpy())
    
    return rgbs, depths

def build_scan_from_msg(lidar_msg: PointCloud2, timestamp: rospy.Time) -> LidarScan:
    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(
        lidar_msg)

    lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
    xyz = lidar_data[:, :3]
    
    dists = torch.linalg.norm(xyz, dim=1)
    valid_ranges = dists > 0

    xyz = xyz[valid_ranges].T
    timestamps = (lidar_data[valid_ranges, -1] + timestamp.to_sec()).float()

    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:, indices]

    feature_mask = torch.zeros(lidar_data.shape[0])

    return LidarScan(directions.float(), dists.float(), feature_mask.float(), timestamps.float())

def build_lidar_rays(lidar_scan: LidarScan,
                    lidar_indices: torch.Tensor,
                    ray_range: torch.Tensor,
                    world_cube: WorldCube,
                    lidar_poses: torch.Tensor, # 4x4
                    ignore_world_cube: bool = False) -> torch.Tensor:
    # _use_simple_frame = True

    rotate_lidar_opengl = torch.eye(4) #.to(self._device)
    rotate_lidar_points_opengl = torch.eye(3) #.to(self._device)

    depths = lidar_scan.distances[lidar_indices] / world_cube.scale_factor
    directions = lidar_scan.ray_directions[:, lidar_indices]
    timestamps = lidar_scan.timestamps[lidar_indices]

    # N x 4 x 4
    print('lidar_poses.shape: ', lidar_poses.shape)
    # Now that we're in OpenGL frame, we can apply world cube transformation
    ray_origins: torch.Tensor = lidar_poses[..., :3, 3]
    ray_origins = ray_origins + world_cube.shift
    ray_origins = ray_origins / world_cube.scale_factor
    ray_origins = ray_origins @ rotate_lidar_opengl[:3,:3]

    ray_origins = ray_origins.tile(len(timestamps), 1)

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

def find_corresponding_lidar_scan(bag, lidar_topic, seq):
    for topic, msg, ts in bag.read_messages(topics=[lidar_topic]):
        if msg.header.seq == seq:
            return msg

def merge_o3d_pc(pcd1, pcd2):
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

def lidar_ts_to_seq(bag, lidar_topic):
    init_ts = -1
    lidar_ts_to_seq = []
    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
        if init_ts == -1:
            init_ts = msg.header.stamp.to_sec() # TBV
        timestamp = msg.header.stamp.to_sec() - init_ts
        lidar_ts_to_seq.append(timestamp)
    return lidar_ts_to_seq

rosbag_path = '/hostroot/home/pckung/fusion_portable/20220216_canteen_day/20220216_canteen_day_ref.bag'
lidar_topic = '/os_cloud_node/points'

mesher = Mesher(model, ckpt, world_cube, rosbag_path, lidar_topic)
mesher.get_mesh(_DEVICE)

# bag = rosbag.Bag(rosbag_path, 'r')
# lidar_ts_to_seq_ = lidar_ts_to_seq(bag, lidar_topic)
# pcd = o3d.geometry.PointCloud()
# with torch.no_grad():
#     poses = ckpt["poses"]    
#     all_poses = []
#     skip_step = 1 #10

#     if args.only_last_frame:
#         tqdm_poses = tqdm([poses[-1]])
#     else:
#         poses = poses[15:]
#         tqdm_poses = tqdm(poses[::skip_step])
#     for i, kf in enumerate(tqdm_poses):
#         if args.use_gt_poses:
#             start_key = "gt_start_lidar_pose"
#             end_key = "gt_end_lidar_pose"
#             pose_key = "gt_lidar_pose"
#         else:
#             start_key = "start_lidar_pose"
#             end_key = "end_lidar_pose"
#             pose_key = "lidar_pose"

#         kf_timestamp = kf["timestamp"].numpy()
#         print(kf_timestamp)

#         seq = np.argmin(np.abs(np.array(lidar_ts_to_seq_) - kf_timestamp))
#         print('seq: ', seq)

#         lidar_msg = find_corresponding_lidar_scan(bag, lidar_topic, seq)
#         lidar_scan = build_scan_from_msg(lidar_msg, lidar_msg.header.stamp).to(_DEVICE)
#         lidar_pose = Pose(pose_tensor=kf[pose_key]).to(_DEVICE)
#         # print('lidar_pose: \n', lidar_pose.get_transformation_matrix())
#         ray_directions = LidarRayDirections(lidar_scan, chunk_size=CHUNK_SIZE, device=_DEVICE)

#         size = lidar_scan.ray_directions.shape[1]
#         rgb_fine = torch.zeros((size,3), dtype=torch.float32).view(-1, 3)
#         depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)
#         for chunk_idx in range(ray_directions.num_chunks):
#             eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, world_cube, ray_range)
#             eval_rays = eval_rays.to(_DEVICE)
#             # print('eval_rays.shape: ', eval_rays.shape)
#             results = model(eval_rays, ray_sampler, scale_factor, testing=True)

#             rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
#             depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)  * scale_factor

#         print('lidar_scan.ray_directions.shape: ', lidar_scan.ray_directions.shape)
#         print('depth_fine.shape: ', depth_fine.shape)

#         depth_gt = torch.unsqueeze(lidar_scan.distances, 1)
#         print('depth_gt.shape: ', depth_gt.shape)

#         rendered_lidar = (lidar_scan.ray_directions.t() * depth_fine).cpu().numpy()
#         print('rgb_fine.shape: ', rgb_fine.shape)
#         rendered_colors = rgb_fine.cpu().numpy()
#         gt_lidar = (lidar_scan.ray_directions.t() * depth_gt).cpu().numpy()

#         rendered_pcd = o3d.geometry.PointCloud()
#         rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar)
#         rendered_pcd.colors = o3d.utility.Vector3dVector(rendered_colors)
#         # rendered_pcd.paint_uniform_color([0, 1, 0.25])
#         lidar_pcd = o3d.geometry.PointCloud()
#         lidar_pcd.points = o3d.utility.Vector3dVector(gt_lidar)
#         lidar_pcd.paint_uniform_color([1, 0, 0.25])

#         pcd = merge_o3d_pc(pcd, rendered_pcd.transform(lidar_pose.get_transformation_matrix().cpu().numpy()))
#         # o3d.visualization.draw_geometries([pcd])
#         o3d.visualization.draw_geometries([rendered_pcd])


        # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        # pcd.estimate_normals()
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        # o3d.visualization.draw_geometries([pcd, mesh])

        # alpha = 0.5
        # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
        # mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])

        # rendered_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        # rendered_pcd.estimate_normals()
        # radii = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4]
        # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     rendered_pcd, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([rec_mesh])

        # import sys
        # sys.exit()

        # if start_key in kf:
        #     start_lidar_pose = Pose(pose_tensor=kf[start_key])
        #     end_lidar_pose = Pose(pose_tensor=kf[end_key])
        #     start_camera_pose = start_lidar_pose * lidar_to_camera
        #     end_camera_pose = end_lidar_pose * lidar_to_camera

        #     start_rendered, start_depth_rendered, _ = render_dataset_frame(start_camera_pose.to(_DEVICE))
        #     end_rendered, end_depth_rendered, _ = render_dataset_frame(end_camera_pose.to(_DEVICE))
            
        #     save_img(start_rendered, [], f"predicted_img_{label}_start.png", render_dir)
        #     save_img(end_rendered, [], f"predicted_img_{label}_end.png", render_dir)
        #     save_depth(start_depth_rendered, f"predicted_depth_{label}_start.png", render_dir)
        #     save_depth(end_depth_rendered, f"predicted_depth_{label}_end.png", render_dir)
          
        # else:
        #     lidar_pose= Pose(pose_tensor=kf[pose_key])
        #     cam_pose = lidar_pose * lidar_to_camera
        #     rgb, depth, _ = render_dataset_frame(cam_pose.to(_DEVICE))
        #     save_img(rgb, [], f"predicted_img_{label}.png", render_dir)
        #     save_depth(depth, f"predicted_depth_{label}.png", render_dir)

