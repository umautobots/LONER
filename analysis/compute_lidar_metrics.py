import argparse
import os
import sys
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import re
import open3d as o3d
import tqdm
import pickle
import yaml
import pathlib
import torch.multiprocessing as mp

CHUNK_SIZE=2**15

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose import Pose
from src.common.sensors import LidarScan
from src.common.ray_utils import LidarRayDirections
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler
from src.common.pose_utils import WorldCube

def load_scan_poses(yaml_path, scan_nums):
    transform_data = cv2.FileStorage(yaml_path, cv2.FileStorage_READ)

    result = {}
    for scan_num in scan_nums:
        t = transform_data.getNode(f"translation_{scan_num}").mat()[0].reshape(-1, 1)
        q = transform_data.getNode(f"quaternion_{scan_num}").mat()[0]

        # swap w,x,y,z for x,y,z,w
        q = np.hstack((q[1],q[2],q[3],q[0]))

        R = Rotation.from_quat(q).as_matrix()

        T = np.hstack((R, t))
        T = np.vstack((T, [0,0,0,1]))
        T = torch.from_numpy(T)

        result[scan_num] = T

    return result

def process_sequence(args, experiment_directory, var_threshold):
    checkpoints = os.listdir(f"{experiment_directory}/checkpoints")

    if args.ckpt_id is None:
        #https://stackoverflow.com/a/2669120
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
    else:
        checkpoint = f"ckpt_{args.ckpt_id}.tar"

    checkpoint_path = pathlib.Path(f"{experiment_directory}/checkpoints/{checkpoint}")

    scan_names = os.listdir(f"{args.groundtruth_map_directory}/scan")
    scan_nums = [s[:-4] for s in scan_names]

    scan_poses = load_scan_poses(f"{args.groundtruth_map_directory}/transformation.yaml", scan_nums)

    groundtruth_traj = args.groundtruth_trajectory
    gt_traj_df = pd.read_csv(groundtruth_traj, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    start_pose = gt_traj_df.to_numpy()[0]

    t = start_pose[1:4]
    q = start_pose[4:]
    R = Rotation.from_quat(q).as_matrix()
    start_pose = np.hstack((R,t.reshape(-1, 1)))
    start_pose = np.vstack((start_pose, [0,0,0,1]))
    T_world_start = torch.from_numpy(start_pose)

    with open(f"{experiment_directory}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)

    _DEVICE = torch.device(full_config.mapper.device)


    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if not checkpoint_path.exists():
        print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
        exit()

    occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
    assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"

    scale_factor = full_config.world_cube.scale_factor.cuda()
    shift = full_config.world_cube.shift
    world_cube = WorldCube(scale_factor, shift).to(0)

    # Returns the 3D logits as a 5D tensor
    occ_model = OccupancyGridModel(occ_model_config).cuda()

    ray_sampler = OccGridRaySampler()

    # use single fine MLP when using OGM
    model_config = full_config.mapper.optimizer.model_config.model
    model = Model(model_config).cuda()


    print(f'Loading checkpoint from: {checkpoint_path}')
    ckpt = torch.load(str(checkpoint_path))

    model.load_state_dict(ckpt['network_state_dict'])

    cfg = full_config.mapper.optimizer.model_config
    ray_range = cfg.data.ray_range


    occ_model.load_state_dict(ckpt['occ_model_state_dict'])
    # initialize occ_sigma
    occupancy_grid = occ_model()
    ray_sampler.update_occ_grid(occupancy_grid.detach())

    with torch.no_grad():
        for scan_num in sorted(scan_nums)[0:1]:
            is_first = True

            T_world_lidar = scan_poses[scan_num]
            T_start_lidar = T_world_start.inverse() @ T_world_lidar

            scan_path = f"{args.groundtruth_map_directory}/scan/{scan_num}.pcd"
            print("Reading scan from", scan_path)
            gt_scan_data = o3d.io.read_point_cloud(scan_path)

            all_points = torch.from_numpy(np.asarray(gt_scan_data.points)).float()

            num_chunks = int(np.ceil(all_points.shape[0] / CHUNK_SIZE))

            depth = torch.zeros((all_points.shape[0],), dtype=torch.float32, device='cpu')
            variance = torch.zeros((all_points.shape[0],), dtype=torch.float32, device='cpu')
            rendered_lidar = torch.zeros((all_points.shape[0], 3,), dtype=torch.float32, device='cpu')
            num_points = 0

            print("Rendering scan with the same intrinsics")
            for chunk_idx in tqdm.tqdm(range(num_chunks)):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = (chunk_idx+1) * CHUNK_SIZE

                points = all_points[start_idx:end_idx].cuda()

                distances = points.norm(dim=1, keepdim=True)
                directions = points/distances

                scan = LidarScan(directions.T, distances, torch.zeros_like(distances))
                
                ray_directions = LidarRayDirections(scan, CHUNK_SIZE)

                size = scan.ray_directions.shape[1]

                lidar_pose = Pose(T_start_lidar).to(0)

                eval_rays = ray_directions.fetch_chunk_rays(0, lidar_pose, world_cube, ray_range)
                eval_rays = eval_rays.cuda()
                results = model(eval_rays, ray_sampler, scale_factor, testing=True, return_variance=True)

                est_variance = results['variance']
                est_depth = results['depth_fine'] * scale_factor

                depth[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE] = est_depth.detach().cpu().clone()

                good_idx = est_variance < var_threshold
                num_excl = (~good_idx).sum()
                good_idx = torch.logical_and(good_idx, est_depth < ray_range[1])
                rendered_lidar[num_points:num_points+good_idx.sum()] = (scan.ray_directions[:,good_idx] * est_depth[good_idx]).cpu().T
                variance[num_points:num_points+good_idx.sum()] = est_variance[good_idx].detach().cpu().clone()

                num_points += good_idx.sum()

            rendered_lidar = rendered_lidar[:num_points]
            variance = variance[:num_points].tile(3, 1).T
            breakpoint()
            rendered_pcd = o3d.geometry.PointCloud()
            rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar.numpy())
            rendered_pcd.colors = o3d.utility.Vector3dVector(torch.clip(variance, 0, 1).numpy())

            print("Estimating point cloud normals")
            rendered_pcd.estimate_normals()
            gt_scan_data.estimate_normals()

            convergence_criteria = (
                o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
                    1e-12,
                    1e-12,
                    10))
            
            print("Refining alignment")
            registration = o3d.pipelines.registration.registration_icp(
                rendered_pcd, gt_scan_data, 0.125, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=convergence_criteria)
            
            registration_result = registration.transformation.copy()
            
            rendered_pcd.transform(registration_result)

            print("Computing metrics")
            accuracy = np.asarray(rendered_pcd.compute_point_cloud_distance(gt_scan_data))
            completion = np.asarray(gt_scan_data.compute_point_cloud_distance(rendered_pcd))
            chamfer_distance = accuracy.mean() + completion.mean()

            # https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/pointcloud.py
            false_negatives = (completion > args.f_score_threshold).sum().item()
            false_positives = (accuracy > args.f_score_threshold).sum().item()
            true_positives = (len(accuracy) - false_positives)

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            stats = {
                "accuracy": accuracy.mean().item(),
                "completion": completion.mean().item(),
                "chamfer_distance": chamfer_distance.item(),
                "recall": recall,
                "precision": precision,
                "f-score": f_score,
                "num_points": len(accuracy)
            }

            metrics_dir = f"{experiment_directory}/metrics"
            renders_dir = f"{experiment_directory}/lidar_renders/"
            os.makedirs(metrics_dir, exist_ok=True)

            if args.write_pointclouds:
                os.makedirs(renders_dir, exist_ok=True
                )
                o3d.io.write_point_cloud(f"{renders_dir}/rendered_{scan_num}_{var_threshold}.pcd", rendered_pcd)

                if is_first:
                    o3d.io.write_point_cloud(f"{renders_dir}/gt_{scan_num}.pcd", gt_scan_data)

                    is_first = False

            with open(f"{metrics_dir}/statistics_{scan_num}_{var_threshold}.yaml", 'w+') as yaml_stats_f:
                yaml.dump(stats, yaml_stats_f, indent = 2)


def _gpu_worker(job_queue: mp.Queue, args, total):
    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            break
        i, exp_dir, var_th = data
        process_sequence(args, exp_dir, var_th)
        print(f"Processed sequence {i+1} of {total}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
    parser.add_argument("experiment_directories", nargs='+', type=str, help="folder in outputs with all results")
    parser.add_argument("groundtruth_map_directory", type=str, help="folder with ground truth map")
    parser.add_argument("groundtruth_trajectory", type=str, help="file with ground truth trajectory")
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--var_threshold", type=float, default = [1e-5], nargs='+', help="Threshold(s) for variance")
    parser.add_argument("--write_pointclouds", default=False, action="store_true")
    parser.add_argument("--f_score_threshold", type=float, default=0.1)
    args = parser.parse_args()

    job_queue = mp.Queue()

    num_jobs = 0
    for exp_dir in args.experiment_directories:
        for var_th in args.var_threshold:
            job_queue.put((num_jobs, exp_dir, var_th,))
            num_jobs += 1

    for _ in range(torch.cuda.device_count()):
        job_queue.put(None)

    gpu_worker_processes = []
    for gpu_id in range(torch.cuda.device_count()):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(job_queue, args, num_jobs)))
        gpu_worker_processes[-1].start()

    # Sync
    for process in gpu_worker_processes:
        process.join()