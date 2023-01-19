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

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from render_utils import *

from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.ray_utils import CameraRayDirections
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler
from src.common.pose_utils import create_spiral_poses

# autopep8: on


assert torch.cuda.is_available(), 'Unable to find GPU'

CHUNK_SIZE=512

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--eval", default=False, dest="eval", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")

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

render_dir = pathlib.Path(f"{args.experiment_directory}/renders")
os.makedirs(render_dir, exist_ok=True)

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

ray_directions = CameraRayDirections(full_config.calibration, chunk_size=CHUNK_SIZE, device=_DEVICE)

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

with torch.no_grad():
    poses = ckpt["poses"]

    all_poses = []
    if args.use_gt_poses:
        start_key = "gt_start_lidar_pose"
        end_key = "gt_end_lidar_pose"
    else:
        start_key = "start_lidar_pose"
        end_key = "end_lidar_pose"

    for kf in tqdm(poses):
        start_lidar_pose = Pose(pose_tensor=kf[start_key])
        end_lidar_pose = Pose(pose_tensor=kf[end_key])
        lidar_to_camera = Pose(pose_tensor=kf["lidar_to_camera"])
        timestamp = kf["timestamp"]

        timestamp = str(timestamp.item()).replace('.','_')[:5]
        
        start_camera_pose = start_lidar_pose * lidar_to_camera
        end_camera_pose = end_lidar_pose * lidar_to_camera

        start_rendered, start_depth_rendered, _ = render_dataset_frame(start_camera_pose.to(_DEVICE))
        end_rendered, end_depth_rendered, _ = render_dataset_frame(end_camera_pose.to(_DEVICE))
        # rgbs, depths = render_spiral(start_camera_pose)

        # rgbs = [torch.from_numpy(rgb) for rgb in rgbs]
        # depths = [torch.from_numpy(depth) for depth in depths]

        
        save_img(start_rendered, [], f"predicted_img_{timestamp}_start.png", render_dir)
        save_img(end_rendered, [], f"predicted_img_{timestamp}_end.png", render_dir)
        save_depth(start_depth_rendered, f"predicted_depth_{timestamp}_start.png", render_dir)
        save_depth(end_depth_rendered, f"predicted_depth_{timestamp}_end.png", render_dir)
        # save_video(f"{render_dir}/spiral_rgb_{timestamp}.gif", rgbs, (int(im_size[0]), int(im_size[1]), 3), rescale=False, clahe=False, isdepth=False, fps=5)
        # save_video(f"{render_dir}/spiral_depth_{timestamp}.gif", depths, (int(im_size[0]), int(im_size[1])), rescale=False, clahe=False, isdepth=True, fps=5)
        