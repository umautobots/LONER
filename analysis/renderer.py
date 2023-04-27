#!/usr/bin/env python
# coding: utf-8

"""
File: renderer.py
Description: Renders RGB and Depth from a saved checkpoint. 

Unless --no_render_stills is set, this will render RGB and color for each of the keyframe poses.

If --render_video is set, it will create a path along the ground truth trajectory, and render
dense frames. 
"""

import argparse
import os
import pathlib
import pickle
import re
import sys
import time
import torch
import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

IM_SCALE_FACTOR = 1

from render_utils import *

from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.ray_utils import CameraRayDirections
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import UniformRaySampler, OccGridRaySampler


assert torch.cuda.is_available(), 'Unable to find GPU'

CHUNK_SIZE=1024

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--eval", default=False, dest="eval", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
parser.add_argument("--no_render_stills", action="store_true", default=False)
parser.add_argument("--render_video", action="store_true", default=False)
parser.add_argument("--skip_step", type=int, default=15, dest="skip_step")
parser.add_argument("--only_last_frame", action="store_true", default=False)
parser.add_argument("--sep_ckpt_result", action="store_true", default=False)
parser.add_argument("--start_frame", type=int, default=0, dest="start_frame")

parser.add_argument("--render_pose", default=None, type=float, nargs=6, help="x y z y p r render pose (angles in degrees).")

args = parser.parse_args()

checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

if args.ckpt_id is None:
    if "final.tar" in checkpoints:
        checkpoint = "final.tar"
    else:
        #https://stackoverflow.com/a/2669120
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"

checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")

if args.sep_ckpt_result:
    render_dir = pathlib.Path(f"{args.experiment_directory}/renders/{checkpoint}_start{args.start_frame}_step{args.skip_step}")
else:
    render_dir = pathlib.Path(f"{args.experiment_directory}/renders")
os.makedirs(render_dir, exist_ok=True)

# override any params loaded from yaml
with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

if full_config["calibration"]["camera_intrinsic"]["width"] is not None:
    full_config["calibration"]["camera_intrinsic"]["width"] *= IM_SCALE_FACTOR
    full_config["calibration"]["camera_intrinsic"]["height"] *= IM_SCALE_FACTOR
    full_config["calibration"]["camera_intrinsic"]["k"] *= IM_SCALE_FACTOR
    full_config["calibration"]["camera_intrinsic"]["new_k"] *= IM_SCALE_FACTOR
else:
    # Sensible defaults for lidar-only case
    full_config["calibration"]["camera_intrinsic"]["width"] = int(1024/2 * IM_SCALE_FACTOR)
    full_config["calibration"]["camera_intrinsic"]["height"] = int(768/2 * IM_SCALE_FACTOR)
    full_config["calibration"]["camera_intrinsic"]["k"] = torch.Tensor([[302,  0.0, 260],
                                                                        [ 0.0, 302, 197],
                                                                        [ 0.0,  0.0, 1.0]]) * IM_SCALE_FACTOR
    full_config["calibration"]["camera_intrinsic"]["new_k"] = full_config["calibration"]["camera_intrinsic"]["k"]
    full_config["calibration"]["camera_intrinsic"]["distortion"] = torch.zeros(4)
    full_config["calibration"]["lidar_to_camera"]["orientation"] = np.array([0.5, -0.5, 0.5, -0.5]) # for weird compatability 


intrinsic = full_config.calibration.camera_intrinsic
im_size = torch.Tensor([intrinsic.height, intrinsic.width])

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

ray_directions = CameraRayDirections(full_config.calibration, chunk_size=CHUNK_SIZE, device=_DEVICE)

# use single fine MLP when using OGM
model_config = full_config.mapper.optimizer.model_config.model
model = Model(model_config).to(_DEVICE)

print(f'Loading checkpoint from: {checkpoint_path}')
ckpt = torch.load(str(checkpoint_path))
model.load_state_dict(ckpt['network_state_dict'])

if full_config.mapper.optimizer.samples_selection.strategy == 'OGM':
    occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
    assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
    # Returns the 3D logits as a 5D tensor
    occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
    ray_sampler = OccGridRaySampler()
    occ_model.load_state_dict(ckpt['occ_model_state_dict'])
    # initialize occ_sigma
    occupancy_grid = occ_model()
    ray_sampler.update_occ_grid(occupancy_grid.detach())
else:
    ray_sampler = UniformRaySampler()

cfg = full_config.mapper.optimizer.model_config
ray_range = cfg.data.ray_range


def render_dataset_frame(pose: Pose):
    with torch.no_grad():
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
            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, pose.clone(), world_cube, ray_range)
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

def _gpu_worker(job_queue: mp.Queue, result_queue: mp.Queue, lidar_to_cam, total):
    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            result_queue.put(None)
            break
        pose_idx, pose = data
        cam_pose = Pose(pose)*lidar_to_cam.to('cpu')
        rgb, depth, _ = render_dataset_frame(cam_pose.to(_DEVICE))
        print(f"Rendered frame {pose_idx}/{total-1}")
        result_queue.put((pose_idx, depth.cpu(), rgb.cpu()))
    
    while True:
        continue
if __name__ == "__main__":
    mp.set_start_method('spawn')

    with torch.no_grad():
        poses = ckpt["poses"]
        lidar_to_camera = Pose.from_settings(full_config.calibration.lidar_to_camera)
        if args.only_last_frame:
            poses_ = [poses[-1]]

        elif args.render_pose is not None:
            xyz = np.array([args.render_pose[:3]]).squeeze(0)
            rpy = np.array([args.render_pose[3:]])

            rotvec = Rotation.from_euler('ZYX', rpy, True).as_rotvec().squeeze(0)
            T = np.hstack((xyz, rotvec))

            poses_ = [{"lidar_pose": torch.from_numpy(T), "timestamp": torch.tensor([0.0])}]

        else:
            poses_ = poses[args.start_frame:]
            poses_ = poses_[::args.skip_step]
        if not args.no_render_stills:
            for kf in tqdm(poses_):
                
                if args.use_gt_poses:
                    pose_key = "gt_lidar_pose"
                else:
                    pose_key = "lidar_pose"


                timestamp = kf["timestamp"]
                timestamp = str(timestamp.item()).replace('.','_')[:5]

                lidar_pose= Pose(pose_tensor=kf[pose_key])
                cam_pose = lidar_pose.to('cpu') * lidar_to_camera.to('cpu')
                rgb, depth, _ = render_dataset_frame(cam_pose.to(_DEVICE))
                save_img(rgb, [], f"predicted_img_{timestamp}.png", render_dir)
                save_depth(depth, f"predicted_depth_{timestamp}.png", render_dir, max_depth=70)

        if args.render_video:

            # Every <this many> meters, do a 360
            SPIN_SPACING_M = 10

            # how long in seconds should the 360 takes
            SPIN_DURATION_S = 15

            # How fast should the fly-through move, in m/s
            CAMERA_VELOCITY = 1

            FPS = 5

            # Get ground truth trajectory
            rosbag_path = pathlib.Path(full_config.dataset_path)
            ground_truth_file = full_config.run_config["groundtruth_traj"]
            ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")

            ground_truth_data = ground_truth_df.to_numpy(dtype=np.float64)

            gt_xyz = ground_truth_data[:,1:4]
            gt_quats = ground_truth_data[:,4:]
            rotations = Rotation.from_quat(gt_quats)

            T = np.concatenate([rotations.as_matrix(), gt_xyz.reshape((-1, 3, 1))], axis=2)
            homog = np.tile(np.asarray([0,0,0,1]), (T.shape[0], 1, 1))
            T = np.concatenate([T, homog], axis=1)
            start_pose = T[0].copy()
            T = np.linalg.inv(start_pose) @ T

            gt_xyz = T[:, :3, 3]
            rotations = Rotation.from_matrix(T[:, :3, :3])

            diffs = np.diff(gt_xyz, axis=0)
            dists = np.sqrt( np.sum(diffs ** 2, axis=1))

            # Normalize the camera velocity
            dts = dists / CAMERA_VELOCITY
            timestamps = np.cumsum(dts)
            timestamps = np.insert(timestamps, 0, 0.)

            slerp = Slerp(timestamps, rotations)
            xyz_interp = interp1d(timestamps, gt_xyz, axis=0)

            num_images = int(timestamps[-1] * FPS)
            image_timestamps = np.linspace(0, timestamps[-1], num_images)

            lidar_poses = []

            dist_since_last_spin = 0
            prev_pose = np.eye(4)
            
            spin_idxs = []
            for timestamp in image_timestamps:

                xyz = xyz_interp(timestamp)
                rot = slerp(timestamp)

                T = np.hstack((rot.as_matrix(), xyz.reshape(-1, 1)))
                T = np.vstack((T, [0,0,0,1]))
                lidar_poses.append(T)
                
                dist_since_last_spin += np.sqrt(np.sum((xyz - prev_pose[:3,3])**2))

                if dist_since_last_spin > SPIN_SPACING_M:
                    num_spin_steps =  SPIN_DURATION_S * FPS
                    spin_amounts_rad = np.linspace(0, 2*np.pi, num_spin_steps)
                    rotations = Rotation.from_euler('z', spin_amounts_rad)
                    
                    
                    for rel_rot in rotations:
                        spin_idxs.append(len(lidar_poses))
                        T = np.hstack((rot.as_matrix() @ rel_rot.as_matrix(), xyz.reshape(-1, 1)))
                        T = np.vstack((T, [0,0,0,1]))
                        spin_idxs.append(len(lidar_poses))

                        lidar_poses.append(T)

                    dist_since_last_spin = 0

                prev_pose = T

            lidar_poses = torch.from_numpy(np.stack(lidar_poses).astype(np.float32))

            # rgbs = []
            depths = []
            job_queue = mp.Queue()
            result_queue = mp.Queue()
            for pose_idx, pose in enumerate(lidar_poses):
                job_queue.put((pose_idx, pose))

            for _ in range(torch.cuda.device_count()):
                job_queue.put(None)

            gpu_worker_processes = []
            for gpu_id in range(torch.cuda.device_count()):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(job_queue, result_queue,lidar_to_camera, len(lidar_poses))))
                gpu_worker_processes[-1].start()


            stop_recv = 0
            results = []
            while stop_recv < torch.cuda.device_count():
                result = result_queue.get()
                if result is None:
                    stop_recv += 1
                    continue
                results.append(result)
            # Sync
            for process in gpu_worker_processes:
                process.terminate()

            results = sorted(results)

            depths = [r[1] for r in results]
            rgbs = [r[2].permute(0,2,3,1).detach() for r in results]

            rgbs_nospin = []
            depths_nospin = []

            for idx, (r,d) in enumerate(zip(rgbs, depths)):
                if idx not in spin_idxs:
                    rgbs_nospin.append(r)
                    depths_nospin.append(d)

            save_video(f"{render_dir}/flythrough_depth.mp4", depths, depths[0].size(), 
                    cmap='turbo', rescale=False, clahe=False, isdepth=True, fps=FPS*6)

            save_video(f"{render_dir}/flythrough_depth_nospin.mp4", depths_nospin, depths[0].size(), 
                    cmap='turbo', rescale=False, clahe=False, isdepth=True, fps=FPS*6)
            
            save_video(f"{render_dir}/flythrough_depth.gif", depths, depths[0].size(), 
                    cmap='turbo', rescale=False, clahe=False, isdepth=True, fps=FPS*6)

            save_video(f"{render_dir}/flythrough_depth_nospin.gif", depths_nospin, depths[0].size(), 
                    cmap='turbo', rescale=False, clahe=False, isdepth=True, fps=FPS*6)            
            
            save_video(f"{render_dir}/flythrough_rgb.mp4", rgbs, rgbs[0].shape, 
                    cmap='turbo', rescale=False, clahe=False, isdepth=False, fps=FPS*6)
                
            save_video(f"{render_dir}/flythrough_rgb_nospin.mp4", rgbs_nospin, rgbs[0].shape, 
                    cmap='turbo', rescale=False, clahe=False, isdepth=False, fps=FPS*6)
                
