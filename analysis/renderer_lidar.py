#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pathlib
import pickle
import re
import sys
import torch
import pandas as pd
import tqdm
import rosbag
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

import open3d as o3d

from render_utils import *

from src.common.pose import Pose
from src.common.pose_utils import WorldCube, build_poses_from_df
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler


from src.common.sensors import LidarScan
from src.common.ray_utils import LidarRayDirections

CHUNK_SIZE=2**12

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


def render_scan(lidar_pose, ray_directions, render_color: bool = False):
    with torch.no_grad():
        size = ray_directions.lidar_scan.ray_directions.shape[1]

        if render_color:
            rgb_fine = torch.zeros((size,3), dtype=torch.float32).view(-1, 3)

        depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)
        variance = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)

        for chunk_idx in range(ray_directions.num_chunks):
            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, world_cube, ray_range)
            eval_rays = eval_rays.to(_DEVICE)
            results = model(eval_rays, ray_sampler, scale_factor, testing=True, return_variance=True, camera=render_color)

            if render_color:
                rgb_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['rgb_fine']
            depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)  * scale_factor
            variance[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['variance'].unsqueeze(1) * scale_factor

        rendered_lidar = (ray_directions.lidar_scan.ray_directions.t() * depth_fine).cpu().numpy()

        good_idx = variance < args.var_threshold
        good_idx = torch.logical_and(good_idx, depth_fine < ray_range[1] - 0.25)
        good_idx = good_idx.squeeze(1).cpu()
        rendered_lidar = rendered_lidar[good_idx]

        if render_color:
            rendered_colors = rgb_fine.cpu().numpy()
            rendered_colors = rendered_colors[good_idx]

            return rendered_lidar, rendered_colors
        
        return rendered_lidar

## Sketchily keeping these outside main guard for multiprocessing reasons
parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

parser.add_argument("--use_traj_est", default=False, action="store_true", help="If set, interpolates the estimated traj instead of using KF poses")
parser.add_argument("--single_threaded", default=False, action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")

parser.add_argument("--traj_interpolation", type=float, default=3., 
                        help="how often to create a render. Only applies if --use_traj_est is set")
parser.add_argument("--skip_step", type=int, default=10,
                        help="skip_step in poses. Only applies if --use_traj_est is not set")

parser.add_argument("--only_last_frame", default=False, dest="only_last_frame", action="store_true")
parser.add_argument("--var_threshold", type=float, default = 1e-2, help="Threshold for variance")
parser.add_argument("--stack_heights", type=float, nargs="+", required=False, default=None,
    help="If provided, will render extra copies of the trajectories at these heights.")
parser.add_argument("--translation_noise", type=float, default=0, help="std dev of noise to apply to pose tranlsations")
parser.add_argument("--voxel_size", type=float, default=None, required=True)
parser.add_argument("--max_range", type=float, default=None)
parser.add_argument("--vert_fov", type=float, nargs=2, default=None)
parser.add_argument("--render_pose", default=None, type=float, nargs=6, help="x y z y p r render pose (angles in degrees).")

args = parser.parse_args()

checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

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

render_dir = pathlib.Path(f"{args.experiment_directory}/renders/{checkpoint}")
os.makedirs(render_dir, exist_ok=True)

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

cfg = full_config.mapper.optimizer.model_config

ray_range = cfg.data.ray_range
if args.max_range is not None:
    ray_range = (ray_range[0], args.max_range)

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

if __name__ == "__main__":
    print(f'Loading checkpoint from: {checkpoint_path}')
ckpt = torch.load(str(checkpoint_path))
model.load_state_dict(ckpt['network_state_dict'])


occ_model.load_state_dict(ckpt['occ_model_state_dict'])
# initialize occ_sigma
occupancy_grid = occ_model()
ray_sampler.update_occ_grid(occupancy_grid.detach())

def _gpu_worker(job_queue, result_queue, ray_directions):
    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            result_queue.put(None)
            break
        
        _, pose = data

        rendered_lidar = render_scan(pose, ray_directions, False)

        result_queue.put((rendered_lidar, pose.clone(),))
    while True:
        continue

if __name__ == "__main__":
    mp.set_start_method('spawn')
   
    os.makedirs(f"{args.experiment_directory}/lidar_renders", exist_ok=True)

    if args.use_traj_est:
        pose_df = pd.read_csv(f"{args.experiment_directory}/trajectory/estimated_trajectory.txt", delimiter=' ', header=None)
        all_poses, _ = build_poses_from_df(pose_df, False)

        translations = all_poses[:, :3, 3]

        current_reference_idx = 0

        render_idxs = []
        for pose_idx in range(len(translations)):
            ref_trans = translations[current_reference_idx]
            current_trans = translations[pose_idx]

            if (current_trans - ref_trans).norm() >= args.traj_interpolation:
                render_idxs.append(pose_idx)
                current_reference_idx = pose_idx

        lidar_poses = all_poses[render_idxs]

        if args.stack_heights is not None:
            all_lidar_poses = []

            stack_heights = list(set([0] + args.stack_heights))

            for height in stack_heights:
                stack = lidar_poses.clone()
                stack[:, 2, 3] += height

                all_lidar_poses.append(stack)

            lidar_poses = torch.cat(all_lidar_poses)
    elif args.render_pose is not None:
        xyz = np.array([args.render_pose[:3]]).squeeze(0)
        rpy = np.array([args.render_pose[3:]])

        rotvec = Rotation.from_euler('ZYX', rpy, True).as_rotvec().squeeze(0)
        T = np.hstack((xyz, rotvec))

        lidar_poses = [{"lidar_pose": torch.from_numpy(T), "timestamp": torch.tensor([0.0])}]
    else:
        poses = ckpt["poses"]    
        all_poses = []
        skip_step = args.skip_step
        
        if args.only_last_frame:
            lidar_poses = [poses[-1]]
        else:
            lidar_poses = poses[::skip_step]

    if args.translation_noise != 0:
        for pose in lidar_poses:
            pose[:3, 3] += torch.normal(torch.zeros_like(pose[:3,3]), args.translation_noise)

    if args.vert_fov is not None:
        vert_fov = args.vert_fov
    else:
        try:
            vert_fov = full_config.run_config.lidar_vertical_fov
        except Exception as e:
            print("Vertical FOV not in config. Please pass as parameter instead")

    lidar_intrinsics = {
        "vertical_fov": vert_fov,
        "vertical_resolution": 0.1,
        "horizontal_resolution": 0.1
    }

    lidar_scan = build_lidar_scan(lidar_intrinsics)
    ray_directions = LidarRayDirections(lidar_scan, chunk_size=CHUNK_SIZE)

    jobs = []
    for pose_idx, pose_state in enumerate(lidar_poses):

        if isinstance(pose_state, dict):
            if args.use_gt_poses:
                pose_key = "gt_lidar_pose"
            else:
                pose_key = "lidar_pose"

            kf_timestamp = pose_state["timestamp"].cpu().numpy()

            lidar_pose = Pose(pose_tensor=pose_state[pose_key]).to(_DEVICE)
        else:
            lidar_pose = Pose(pose_state).to(_DEVICE)
            
        jobs.append((pose_idx, lidar_pose,))

    output_cloud = o3d.geometry.PointCloud()

    if args.single_threaded:
        for _, lidar_pose in tqdm(jobs):
            rendered_lidar = render_scan(lidar_pose, ray_directions, False)

            rendered_pcd = o3d.geometry.PointCloud()
            rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar)

            if args.voxel_size is not None:
                rendered_pcd = rendered_pcd.voxel_down_sample(args.voxel_size)

            output_cloud = merge_o3d_pc(output_cloud, rendered_pcd.transform(lidar_pose.get_transformation_matrix().cpu().numpy()))
    else:
        job_queue = mp.Queue()

        for job in jobs:
            job_queue.put(job)

        for _ in range(torch.cuda.device_count()):
            job_queue.put(None)

        result_queue = mp.Queue()

        gpu_worker_processes = []
        for gpu_id in range(torch.cuda.device_count()):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(job_queue, result_queue, ray_directions,)))
            gpu_worker_processes[-1].start()


        stop_recv = 0        
        pbar = tqdm(total=len(lidar_poses))
        while stop_recv < torch.cuda.device_count():
            result = result_queue.get()
            if result is None:
                stop_recv += 1
                continue
            
            pbar.update(1)

            rendered_lidar, lidar_pose = result

            rendered_pcd = o3d.geometry.PointCloud()
            rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar)
            
            if args.voxel_size is not None:
                rendered_pcd = rendered_pcd.voxel_down_sample(args.voxel_size)

            output_cloud = merge_o3d_pc(output_cloud, rendered_pcd.transform(lidar_pose.get_transformation_matrix().cpu().numpy()))
        
        # Sync
        for process in gpu_worker_processes:
            process.terminate()

    if args.voxel_size is not None:
        output_cloud = output_cloud.voxel_down_sample(args.voxel_size)
    o3d.io.write_point_cloud(f"{args.experiment_directory}/lidar_renders/render_full_{args.voxel_size}.pcd", output_cloud)
