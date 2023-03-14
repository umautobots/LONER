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

# autopep8: on

from src.common.sensors import Image, LidarScan
from sensor_msgs.msg import Image, PointCloud2
import pandas as pd
import ros_numpy
from src.common.lidar_ray_utils import LidarRayDirections, get_far_val
from src.common.mesher import Mesher

assert torch.cuda.is_available(), 'Unable to find GPU'

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--resolution", type=float, default=0.1)
parser.add_argument("--color", default=False, dest="color", action="store_true")
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
args = parser.parse_args()
checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

# meshing_bound = [[-35,25], [-30,45], [-3,15]] # canteen
meshing_bound = [[-16,10], [-6,5], [-3,3]] # mcr
resolution = args.resolution
sigma_only = not args.color
use_lidar_fov_mask = True
use_convex_hull_mask = False

rosbag_path = '/hostroot/home/pckung/fusion_portable/20220216_canteen_day/20220216_canteen_day_ref.bag'
lidar_topic = '/os_cloud_node/points'

# if not os.path.exists(f"{args.experiment_directory}/meshing"):
#     os.makedirs(f"{args.experiment_directory}/meshing")
# mesh_out_file=f"{args.experiment_directory}/meshing/meshing_ckpt_{args.ckpt_id}_res_{resolution}.ply"

mesh_out_file="/hostroot/home/pckung/meshing_mcr_res0.02.ply"

if args.ckpt_id is None:
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
elif args.ckpt_id=='final':
    checkpoint = f"final.tar"
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"

checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")
# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)
if args.debug:
    full_config['debug'] = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
_DEVICE = torch.device(full_config.mapper.device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()

scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
shift = full_config.world_cube.shift
world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

# use single fine MLP when using OGM
model_config = full_config.mapper.optimizer.model_config.model
model = Model(model_config).to(_DEVICE)

print(f'Loading checkpoint from: {checkpoint_path}') 
ckpt = torch.load(str(checkpoint_path)) 
model.load_state_dict(ckpt['network_state_dict']) 

mesher = Mesher(model, ckpt, world_cube, rosbag_path, lidar_topic, resolution=resolution, marching_cubes_bound=meshing_bound, points_batch_size=500000)
mesh_o3d, mesh_lidar_frames, origin_frame = mesher.get_mesh(_DEVICE, sigma_only=sigma_only, threshold=0, 
                                                            use_lidar_fov_mask=use_lidar_fov_mask, use_convex_hull_mask=use_convex_hull_mask)
# mesh post-processing
# mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=1)
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d, origin_frame],
                                    mesh_show_back_face=True, mesh_show_wireframe=False)


print('mesh_out_file: ', mesh_out_file)
o3d.io.write_triangle_mesh(mesh_out_file, mesh_o3d, compressed=False, write_vertex_colors=True, 
                           write_triangle_uvs=False, print_progress=True)
