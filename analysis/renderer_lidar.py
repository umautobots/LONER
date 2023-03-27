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
from src.common.ray_utils import points_to_pcd
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler

# autopep8: on

from src.common.sensors import Image, LidarScan
from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
from src.common.ray_utils import LidarRayDirections, get_far_val

assert torch.cuda.is_available(), 'Unable to find GPU'

CHUNK_SIZE=512

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--eval", default=False, dest="eval", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
parser.add_argument("--only_last_frame", default=False, dest="only_last_frame", action="store_true")
parser.add_argument("--var_threshold", type=float, default = 1e-4, help="Threshold for variance")
parser.add_argument("--write_intermediate_clouds", default=False, action="store_true")

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

render_dir = pathlib.Path(f"{args.experiment_directory}/renders/{checkpoint}_gt_poses_{args.use_gt_poses}")
os.makedirs(render_dir, exist_ok=True)

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)


torch.backends.cudnn.enabled = True


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

    return LidarScan(directions.float(), dists.float(), timestamps.float())



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

rosbag_path = full_config.dataset_path
lidar_topic = '/os_cloud_node/points'

bag = rosbag.Bag(rosbag_path, 'r')
lidar_ts_to_seq_ = lidar_ts_to_seq(bag, lidar_topic)

os.makedirs(f"{args.experiment_directory}/lidar_renders", exist_ok=True)

pcd = o3d.geometry.PointCloud()
with torch.no_grad():
    poses = ckpt["poses"]    
    all_poses = []
    skip_step = 1 #10

    if args.only_last_frame:
        tqdm_poses = tqdm([poses[-1]])
    else:
        tqdm_poses = tqdm(poses[::skip_step])
    for pose_idx, keyframe in enumerate(tqdm_poses):
        if args.use_gt_poses:
            start_key = "gt_start_lidar_pose"
            end_key = "gt_end_lidar_pose"
            pose_key = "gt_lidar_pose"
        else:
            start_key = "start_lidar_pose"
            end_key = "end_lidar_pose"
            pose_key = "lidar_pose"

        kf_timestamp = keyframe["timestamp"].numpy()

        seq = np.argmin(np.abs(np.array(lidar_ts_to_seq_) - kf_timestamp))

        lidar_msg = find_corresponding_lidar_scan(bag, lidar_topic, seq)
        lidar_scan = build_scan_from_msg(lidar_msg, lidar_msg.header.stamp).to(_DEVICE)
        lidar_pose = Pose(pose_tensor=keyframe[pose_key]).to(_DEVICE)
        ray_directions = LidarRayDirections(lidar_scan, chunk_size=CHUNK_SIZE)

        size = lidar_scan.ray_directions.shape[1]
        rgb_fine = torch.zeros((size,3), dtype=torch.float32).view(-1, 3)
        depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)
        variance = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)
        for chunk_idx in range(ray_directions.num_chunks):
            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, world_cube, ray_range)
            eval_rays = eval_rays.to(_DEVICE)
            results = model(eval_rays, ray_sampler, scale_factor, testing=True, return_variance=True)

            rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
            depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)  * scale_factor
            variance[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['variance'].unsqueeze(1)

        depth_gt = torch.unsqueeze(lidar_scan.distances, 1)

        rendered_lidar = (lidar_scan.ray_directions.t() * depth_fine).cpu().numpy()
        rendered_colors = rgb_fine.cpu().numpy()
        gt_lidar = (lidar_scan.ray_directions.t() * depth_gt).cpu().numpy()


        good_idx = variance < args.var_threshold
        good_idx = good_idx.squeeze(1).cpu()
        rendered_lidar = rendered_lidar[good_idx]
        rendered_colors = rendered_colors[good_idx]

        rendered_pcd = o3d.geometry.PointCloud()
        rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar)
        rendered_pcd.colors = o3d.utility.Vector3dVector(rendered_colors)

        gt_lidar_pcd = o3d.geometry.PointCloud()
        gt_lidar_pcd.points = o3d.utility.Vector3dVector(gt_lidar)
        gt_lidar_pcd.paint_uniform_color([1, 0, 0.25])

        pcd = merge_o3d_pc(pcd, rendered_pcd.transform(lidar_pose.get_transformation_matrix().cpu().numpy()))


        if args.write_intermediate_clouds and pose_idx % 10 == 0:
            o3d.io.write_point_cloud(f"{args.experiment_directory}/lidar_renders/render_{pose_idx}.pcd", pcd)

    o3d.io.write_point_cloud(f"{args.experiment_directory}/lidar_renders/render_full.pcd", pcd)