import open3d as o3d
import numpy as np
import os, sys
import yaml
import argparse
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)

from src.common.pose_utils import build_poses_from_df

def compare_point_clouds(est_scan, gt_scan, output_dir, f_score_threshold, voxel_size = 0.05,
                         write_pointclouds = False, write_gt_cloud = False, id_str = None):
    
    print("Downsampling clouds to voxel size", voxel_size)
    est_scan = est_scan.voxel_down_sample(voxel_size)
    gt_scan = gt_scan.voxel_down_sample(voxel_size)

    input_est_size = len(est_scan.points)
    input_gt_size = len(gt_scan.points)

    gt_downsample_skip, est_downsample_skip = 1,1
    if input_est_size > 1_000_000:
        est_downsample_skip = int(input_est_size / 1_000_000)
    if input_gt_size > 1_000_000:
        gt_downsample_skip = int(input_gt_size / 1_000_000)

    alignment_gt_cloud = gt_scan.uniform_down_sample(gt_downsample_skip)
    alignment_est_cloud = est_scan.uniform_down_sample(est_downsample_skip)

    print("Estimating point cloud normals")
    alignment_gt_cloud.estimate_normals()
    alignment_est_cloud.estimate_normals()

    convergence_criteria = (
        o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
            1e-12,
            1e-12,
            10))
    
    print("Refining alignment")
    registration = o3d.pipelines.registration.registration_icp(
        alignment_est_cloud, alignment_gt_cloud, 0.125, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=convergence_criteria)
    
    registration_result = registration.transformation.copy()
    
    est_scan.transform(registration_result)

    o3d.io.write_point_cloud("est_align.pcd", est_scan)
    o3d.io.write_point_cloud("gt_align.pcd", gt_scan)

    print("Computing metrics")
    accuracy = np.asarray(est_scan.compute_point_cloud_distance(gt_scan))
    completion = np.asarray(gt_scan.compute_point_cloud_distance(est_scan))
    chamfer_distance = accuracy.mean() + completion.mean()

    # https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/pointcloud.py
    false_negatives = (completion > f_score_threshold).sum().item()
    false_positives = (accuracy > f_score_threshold).sum().item()
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

    metrics_dir = f"{output_dir}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    id_suffix = f"_{id_str}" if id_str is not None else ""

    if write_pointclouds:
        renders_dir = f"{output_dir}/lidar_renders/"
        os.makedirs(renders_dir, exist_ok=True)

        o3d.io.write_point_cloud(f"{renders_dir}/rendered{id_suffix}.pcd", est_scan)

        if write_gt_cloud:
            o3d.io.write_point_cloud(f"{renders_dir}/gt{id_suffix}.pcd", gt_scan)

    with open(f"{metrics_dir}/statistics{id_suffix}.yaml", 'w+') as yaml_stats_f:
        yaml.dump(stats, yaml_stats_f, indent = 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
    parser.add_argument("experiment_directory", type=str)
    parser.add_argument("gt_map", type=str, help="file with ground truth map")
    parser.add_argument("--gt_trajectory", type=str, default=None, required=False,
        help="If provided, is used to rough align the pointclouds (recommended)")
    parser.add_argument("--estimated_map", default=None, type=str, 
        help="path to estimated map. Defaults to experiment_directory/lidar_renders/render_full.pcd")
    parser.add_argument("--f_score_threshold", type=float, default=0.1)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--initial_transform", default=None, type=float, nargs=16, required=False, help="Initial guess of alignment")
    parser.add_argument("--est_traj", default=None)
    args = parser.parse_args()

    if args.estimated_map is None:
        est_map_path = f"{args.experiment_directory}/lidar_renders/render_full.pcd"
    else:
        est_map_path = f"{args.experiment_directory}/{args.estimated_map}"



    if args.gt_trajectory is None and args.initial_transform is None:
        print("Warning: No GT trajectory provided. Can't rough align maps")
        start_pose = torch.eye(4)
    elif args.initial_transform is not None:
        print("Using supplied initial guess to rough-align clouds")
        start_pose = torch.tensor(args.initial_transform).reshape(4,4)
    else:
        print("Using GT Trajectory to rough-align clouds")
        df = pd.read_csv(args.gt_trajectory, delimiter=' ' , header=None)
        start_pose = build_poses_from_df(df, False)[0][0]

    est_map = o3d.io.read_point_cloud(est_map_path)
    gt_map = o3d.io.read_point_cloud(args.gt_map)

    if args.est_traj is not None:
        df = pd.read_csv(args.est_traj, delimiter=' ' , header=None)
        start_est_pose = build_poses_from_df(df, False)[0][0]

        est_map.transform(start_est_pose.inverse().cpu().numpy())

    gt_map.transform(start_pose.inverse().cpu().numpy())    


    #trial_num = est_map_path[-10:].split("_")[1][0]
    #print(trial_num)
    compare_point_clouds(est_map, gt_map, args.experiment_directory, args.f_score_threshold, args.voxel_size)#, id_str = trial_num)
