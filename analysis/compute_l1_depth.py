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
import torch.nn.functional

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
from examples.run_rosbag import build_scan_from_msg

CHUNK_SIZE=2**12

np.random.seed(0)

def compute_l1_depth(lidar_pose, ray_directions: LidarRayDirections, model_data, render_color: bool = False):
    with torch.no_grad():
        model, ray_sampler, world_cube, ray_range, device = model_data

        scale_factor = world_cube.scale_factor

        size = ray_directions.lidar_scan.ray_directions.shape[1]

        depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)

        for chunk_idx in range(ray_directions.num_chunks):
            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, world_cube, ray_range)
            eval_rays = eval_rays.to(device)
            results = model(eval_rays, ray_sampler, scale_factor, testing=True, return_variance=True, camera=render_color)

            depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)  * scale_factor

        gt_depth = ray_directions.lidar_scan.distances
        good_idx = torch.logical_and(gt_depth.flatten() > ray_range[0], gt_depth.flatten() < ray_range[1] - 0.25) 
        good_depth = depth_fine[good_idx]
        good_gt_depth = gt_depth[good_idx.flatten()]
    
        return torch.nn.functional.l1_loss(good_depth.cpu().flatten(), good_gt_depth.cpu().flatten())

def _gpu_worker(job_queue, result_queue, model_data):
    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            result_queue.put(None)
            break
        
        _, pose, ray_directions = data

        l1 = compute_l1_depth(pose, ray_directions, model_data, False)

        result_queue.put((l1, pose.clone(),))
    while True:
        continue

# We're only going to open the bag once
bag = None

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
    parser.add_argument("experiment_directory", nargs="+", type=str, help="folder in outputs with all results")

    parser.add_argument("--single_threaded", default=False, action="store_true")
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--use_est_poses", action='store_true', default=False)


    args = parser.parse_args()

    for exp_dir in args.experiment_directory:

        checkpoints = os.listdir(f"{exp_dir}/checkpoints")

        if args.ckpt_id is None:
            #https://stackoverflow.com/a/2669120
            convert = lambda text: int(text) if text.isdigit() else text 
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
        elif args.ckpt_id=='final':
            checkpoint = f"final.tar"
        else:
            checkpoint = f"ckpt_{args.ckpt_id}.tar"

        checkpoint_path = pathlib.Path(f"{exp_dir}/checkpoints/{checkpoint}")


        # override any params loaded from yaml
        with open(f"{exp_dir}/full_config.pkl", 'rb') as f:
            full_config = pickle.load(f)

        cfg = full_config.mapper.optimizer.model_config

        ray_range = cfg.data.ray_range

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


        occ_model.load_state_dict(ckpt['occ_model_state_dict'])
        # initialize occ_sigma
        occupancy_grid = occ_model()
        ray_sampler.update_occ_grid(occupancy_grid.detach())

        if bag is None:
            rosbag_path = full_config.dataset_path

            print("Opening RosBag")
            bag = rosbag.Bag(rosbag_path)
            lidar_msgs, lidar_timestamps = [], []

        # Re-creating logic to find init time. should really just log it. TODO. 
        gt_traj_df = pd.read_csv(full_config.run_config.groundtruth_traj, header=None, delimiter=" ")
        gt_poses, gt_timestamps = build_poses_from_df(gt_traj_df, True)

        est_df = pd.read_csv(f"{exp_dir}/trajectory/estimated_trajectory.txt", header=None, delimiter=" ")
        est_poses, est_timestamps = build_poses_from_df(est_df, True)

        init = False
        lidar_topic = f"/{full_config.system.ros_names.lidar}"
        for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
            if not init and timestamp.to_sec() > gt_timestamps[0]:
                start_time = timestamp
                init = True

            lidar_timestamps.append(timestamp)
            lidar_msgs.append(msg)
        
        selected_lidar_indices = np.random.choice(len(est_timestamps), (args.num_frames,), replace=False)

        selected_lidar_msgs = [lidar_msgs[i] for i in selected_lidar_indices]
        selected_lidar_times = [(lidar_timestamps[i] - start_time).to_sec() for i in selected_lidar_indices]
        selected_lidar_times = torch.tensor(selected_lidar_times, dtype=torch.float64)

        if args.use_est_poses:
            all_lidar_timestamps = est_timestamps # est timestamps are strat from zero
            all_lidar_poses = est_poses
        else:

            all_lidar_timestamps = gt_timestamps - start_time.to_sec()  # gt timestamps are not strat from zero
            all_lidar_poses = gt_poses

        abs_diff = torch.abs(all_lidar_timestamps.cpu().unsqueeze(1) - selected_lidar_times.cpu().unsqueeze(0))
        pose_indices = torch.argmin(abs_diff, dim=0)
        lidar_poses = [all_lidar_poses[i] for i in pose_indices]

        jobs = []
        for pose_idx, (pose_state, lidar_msg) in enumerate(zip(lidar_poses, selected_lidar_msgs)):

            if isinstance(pose_state, dict):
                if args.use_gt_poses:
                    pose_key = "gt_lidar_pose"
                else:
                    pose_key = "lidar_pose"

                kf_timestamp = pose_state["timestamp"].numpy()

                lidar_pose = Pose(pose_tensor=pose_state[pose_key]).to(_DEVICE)
            else:
                lidar_pose = Pose(pose_state).to(_DEVICE)

            lidar_scan = build_scan_from_msg(lidar_msg, lidar_msg.header.stamp).to(0)
            ray_directions = LidarRayDirections(lidar_scan, CHUNK_SIZE)
                
            jobs.append((pose_idx, lidar_pose, ray_directions,))

        l1s = []
        if args.single_threaded:
            model_data = (model, ray_sampler, world_cube, ray_range, _DEVICE)
            for _, lidar_pose, ray_directions in tqdm(jobs):
                l1 = compute_l1_depth(lidar_pose, ray_directions, model_data, False)
                l1s.append(l1)
        else:
            job_queue = mp.Queue()

            for job in jobs:
                job_queue.put(job)

            for _ in range(torch.cuda.device_count()):
                job_queue.put(None)

            result_queue = mp.Queue()

            gpu_worker_processes = []
            model_data = (model, ray_sampler, world_cube, ray_range, _DEVICE)
            for gpu_id in range(torch.cuda.device_count()):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(job_queue, result_queue,model_data,)))
                gpu_worker_processes[-1].start()


            stop_recv = 0        
            pbar = tqdm(total=len(lidar_poses))
            while stop_recv < torch.cuda.device_count():
                result = result_queue.get()
                if result is None:
                    stop_recv += 1
                    continue
                l1s.append(result[0])
                pbar.update(1)

            # Sync
            for process in gpu_worker_processes:
                process.terminate()
                process.join()
        
        l1s = torch.hstack(l1s)

        results_dir = f"{exp_dir}/metrics/"
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/l1.yaml", 'w+') as f:
            f.write(f"min: {l1s.min()}\nmax: {l1s.max()}\nmean: {l1s.mean()}\nrmse: {torch.sqrt(torch.mean(l1s**2))}")
