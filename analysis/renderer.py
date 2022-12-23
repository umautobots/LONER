#!/usr/bin/env python
# coding: utf-8

import sys
import os
import torch
import time
import pathlib
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from torchmetrics import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pickle 

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from src.models.model_tcnn import Model, OccupancyGridModel
from src.common.ray_utils import CameraRayDirections
from src.models.ray_sampling import OccGridRaySampler
from src.models.losses import *
from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from render_utils import *

# autopep8: on


assert torch.cuda.is_available(), 'Unable to find GPU'

CHUNK_SIZE=512

print("START")
parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("--experiment_directory", type=str, help="folder in outputs with all results", required=True)

parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--eval", default=False, dest="eval", action="store_true")

args = parser.parse_args()

checkpoint = ""
for ckpt_id in os.listdir(f"{args.experiment_directory}/checkpoints"):
    checkpoint = max(checkpoint, ckpt_id)

checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")

render_dir = f"{args.experiment_directory}/renders"
os.makedirs(render_dir, exist_ok=True)

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

intrinsic = full_config.calibration.camera_intrinsic
im_size = torch.Tensor([intrinsic.height, intrinsic.width])
scale_factor = full_config.world_cube.scale_factor
shift = full_config.world_cube.shift
world_cube = WorldCube(scale_factor, shift)

if args.debug:
    full_config['debug'] = True

ray_directions = CameraRayDirections(full_config.calibration, chunk_size=CHUNK_SIZE)

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

# Load weights checkpoint
if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()

occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"

print('Using Occupancy Grid Model')
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
    rgb_fine = torch.zeros(im_size[0], im_size[1],1, dtype=torch.float32).view(-1, 1)
    depth_fine = torch.zeros(im_size[0],im_size, 1, dtype=torch.float32).view(-1, 1)
    peak_depth_consistency = torch.zeros(im_size[0], im_size[1], 1, dtype=torch.float32).view(-1, 1)
    print("--------------------")
    print("render_dataset_frame")
    for chunk_idx in range(ray_directions.num_chunks):
        print("chunk_idx: ", chunk_idx)
        # tic = time.time()
        batch_eval = ray_directions.fetch_chunk_rays(chunk_idx, pose, world_cube, ray_range)
        eval_rays = batch_eval['rays'].to(_DEVICE)

        results = model(eval_rays, ray_sampler, scale_factor, testing=True)

        rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
        depth = results['depth_fine'].unsqueeze(1)
        depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)

        s_vals = results['samples_fine']
        weights_pred = results['weights_fine']
        s_peaks = s_vals[torch.arange(eval_rays.shape[0]), weights_pred.argmax(dim=1)].unsqueeze(1)
        peak_depth_consistency[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = torch.abs(s_peaks - depth)

    rgb_fine = rgb_fine.reshape(1, im_size[0] , im_size[1] , im_size[2]).permute(0, 3, 1, 2)
    depth_fine = depth_fine.reshape(1, im_size[0], im_size[1] , 1).permute(0, 3, 1, 2) * scale_factor
    peak_depth_consistency = peak_depth_consistency.reshape(1, im_size[0], im_size[1], 1).permute(0, 3, 1, 2) * scale_factor
    print(f'Took: {time.time() - tic_img} seconds for rendering an image')
    
    rgb_fine[..., :cfg.data.filter_top_rows, :] = 1
    depth_fine[..., :cfg.data.filter_top_rows, :] = cfg.data.ray_range[1]
    peak_depth_consistency[..., :cfg.data.filter_top_rows, :] = 0
    
    return rgb_fine.clamp(0, 1), depth_fine, peak_depth_consistency

def save_img(rgb_fine, mask, filename, equalize=False):
    img = rgb_fine.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    if equalize:
        eq_img = equalize_adapthist(img, clip_limit=1.0) * 255
    else:
        eq_img = img * 255
    eq_img[mask, :] = 255
    out_fname = render_dir / f'{filename}'
    imageio.imwrite(str(out_fname), eq_img)

def save_depth(depth_fine, i, min_depth=1, max_depth=50):
    img = depth_fine.squeeze().detach().cpu().numpy()
    mask = (img >= 50)
    img = np.clip(img, min_depth, max_depth)
    # img = (img - img.min()) / (np.percentile(img, 99) - img.min())
    img = (img - min_depth) / (max_depth - min_depth)
    img = np.clip(img, 0, 1)
    cmap = plt.cm.get_cmap('turbo')
    img_colored = cmap(img)
    out_fname = render_dir / f'predicted_depth_{i}.png'
    imageio.imwrite(str(out_fname), img_colored * 255)

    img = mask.astype(float)
    out_fname = render_dir / f'predicted_mask_{i}.png'
    imageio.imwrite(str(out_fname), img * 255)
    return mask

psnr = PeakSignalNoiseRatio()
mssim = MultiScaleStructuralSimilarityIndexMeasure()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

with torch.no_grad():
    print(ckpt)
    poses = ckpt["poses"]

    for kf in poses:
        start_lidar_pose = Pose(pose_tensor=kf["start_lidar_pose"])
        end_lidar_pose = Pose(pose_tensor=kf["end_lidar_pose"])
        lidar_to_camera = Pose(pose_tensor=kf["lidar_to_camera"])
        timestamp = kf["timestamp"]
        
        start_camera_pose = start_lidar_pose * lidar_to_camera
        end_camera_pose = end_lidar_pose * lidar_to_camera

        start_rendered, _, _ = render_dataset_frame(start_camera_pose)
        end_rendered, _, _ = render_dataset_frame(end_camera_pose)
        
        save_img(start_rendered, [], f"predicted_img_{timestamp}_start.png")
        save_img(end_rendered, [], f"predicted_img_{timestamp}_end.png")