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


from src.common.sensors import Image, LidarScan
from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
from src.common.lidar_ray_utils import LidarRayDirections, get_far_val
from analysis.mesher import Mesher

assert torch.cuda.is_available(), 'Unable to find GPU'
import yaml
from pathlib import Path

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
parser.add_argument("configuration_path")
# parser.add_argument("sequence", type=str, default="canteen", help="sequence name. Used to decide meshing bound [canteen | mcr]")
parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--resolution", type=float, default=0.1, help="grid resolution (m)")
parser.add_argument("--threshold", type=float, default=0.0, help="threshold for sigma MLP output. default as 0")

parser.add_argument("--color", default=False, dest="color", action="store_true")
parser.add_argument("--viz", default=False, dest="viz", action="store_true")
parser.add_argument("--save", default=False, dest="save", action="store_true")
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")

parser.add_argument("--use_lidar_fov_mask", default=False, dest="use_lidar_fov_mask", action="store_true")
parser.add_argument("--use_convex_hull_mask", default=False, dest="use_convex_hull_mask", action="store_true")
parser.add_argument("--use_lidar_pointcloud_mask", default=False, dest="use_lidar_pointcloud_mask", action="store_true")
parser.add_argument("--use_occ_mask", default=False, dest="use_occ_mask", action="store_true")

parser.add_argument("--color_render_from_ray", default=False, dest="color_render_from_ray", action="store_true")

args = parser.parse_args()
checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

with open(args.configuration_path) as config_file:
    config = yaml.full_load(config_file)
rosbag_path = Path(os.path.expanduser(config["dataset"]))

x_min, x_max = config['meshing_bounding_box']['x']
y_min, y_max = config['meshing_bounding_box']['y']
z_min, z_max = config['meshing_bounding_box']['z']
meshing_bound = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

resolution = args.resolution
sigma_only = not args.color

use_lidar_fov_mask = args.use_lidar_fov_mask
use_convex_hull_mask = args.use_convex_hull_mask
use_lidar_pointcloud_mask = args.use_lidar_pointcloud_mask
use_occ_mask = args.use_occ_mask
threshold = args.threshold

if args.color_render_from_ray:
    color_mesh_extraction_method = 'render_ray'
else:
    color_mesh_extraction_method = 'direct_point_query'

if args.ckpt_id is None:
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
    args.ckpt_id = checkpoint.split('.')[0]
elif args.ckpt_id=='final':
    checkpoint = f"final.tar"
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"
checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")

if not os.path.exists(f"{args.experiment_directory}/meshing"):
    os.makedirs(f"{args.experiment_directory}/meshing")
mesh_out_file=f"{args.experiment_directory}/meshing/meshing_ckpt_{args.ckpt_id}_res_{resolution}.ply"

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)
if args.debug:
    full_config['debug'] = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
_DEVICE = torch.device(full_config.mapper.device)
print('_DEVICE', _DEVICE)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()

occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
shift = full_config.world_cube.shift
world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

# use single fine MLP when using OGM
model_config = full_config.mapper.optimizer.model_config.model
model = Model(model_config).to(_DEVICE)

print(f'Loading checkpoint from: {checkpoint_path}') 
ckpt = torch.load(str(checkpoint_path))

occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
occ_model.load_state_dict(ckpt['occ_model_state_dict'])
occupancy_grid = occ_model()
ray_sampler = OccGridRaySampler()
ray_sampler.update_occ_grid(occupancy_grid.detach())

model.load_state_dict(ckpt['network_state_dict']) 

# rosbag_path = full_config.dataset_path
lidar_topic = full_config.ros_names.lidar
ray_range = full_config.mapper.optimizer.model_config.data.ray_range
mesher = Mesher(model, ckpt, world_cube, rosbag_path=rosbag_path, lidar_topic=lidar_topic,  resolution=resolution, marching_cubes_bound=meshing_bound, points_batch_size=500000)
mesh_o3d, mesh_lidar_frames = mesher.get_mesh(_DEVICE, ray_sampler, occupancy_grid, occ_voxel_size=occ_model_config.voxel_size, sigma_only=sigma_only, threshold=threshold, 
                                                            use_lidar_fov_mask=use_lidar_fov_mask, use_convex_hull_mask=use_convex_hull_mask,use_lidar_pointcloud_mask=use_lidar_pointcloud_mask, use_occ_mask=use_occ_mask,
                                                            color_mesh_extraction_method=color_mesh_extraction_method)
mesh_o3d.compute_vertex_normals()

if args.viz:
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.squeeze(shift))
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    o3d.visualization.draw_geometries([mesh_o3d, origin_frame],
                                    mesh_show_back_face=True, mesh_show_wireframe=False)

if args.save:
    print('store mesh at: ', mesh_out_file)
    o3d.io.write_triangle_mesh(mesh_out_file, mesh_o3d, compressed=False, write_vertex_colors=True, 
                            write_triangle_uvs=False, print_progress=True)
