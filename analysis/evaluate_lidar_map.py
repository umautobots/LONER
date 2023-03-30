import open3d as o3d
import numpy as np
import os
import yaml
import argparse


def compare_point_clouds(est_scan, gt_scan, ouptput_dir, f_score_threshold,
                         write_pointclouds, write_gt_cloud, id_str = None):

    print("Estimating point cloud normals")
    est_scan.estimate_normals()
    gt_scan.estimate_normals()

    convergence_criteria = (
        o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
            1e-12,
            1e-12,
            10))
    
    print("Refining alignment")
    registration = o3d.pipelines.registration.registration_icp(
        est_scan, gt_scan, 0.125, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=convergence_criteria)
    
    registration_result = registration.transformation.copy()
    
    est_scan.transform(registration_result)

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

    metrics_dir = f"{ouptput_dir}/metrics"
    renders_dir = f"{ouptput_dir}/lidar_renders/"
    os.makedirs(metrics_dir, exist_ok=True)

    if write_pointclouds:
        os.makedirs(renders_dir, exist_ok=True
        )

        id_suffix = f"_{id_str}" if id_str is not None else ""

        o3d.io.write_point_cloud(f"{renders_dir}/rendered{id_suffix}.pcd", est_scan)

        if write_gt_cloud:
            o3d.io.write_point_cloud(f"{renders_dir}/gt{id_suffix}.pcd", gt_scan)

    with open(f"{metrics_dir}/statistics{id_suffix}.yaml", 'w+') as yaml_stats_f:
        yaml.dump(stats, yaml_stats_f, indent = 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
    parser.add_argument("estimated_map", nargs='+', type=str, help="folder in outputs with all results")
    parser.add_argument("gt_map", type=str, help="folder with ground truth map")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--var_threshold", type=float, default = [1e-5], nargs='+', help="Threshold(s) for variance")
    parser.add_argument("--f_score_threshold", type=float, default=0.1)
    args = parser.parse_args()


    est_map = o3d.io.read_point_cloud(args.estimated_map)
    gt_map = o3d.io.read_point_cloud(args.gt_map)

    compare_point_clouds(est_map, gt_map, args.output_dir, args.f_score_threshold, False, False, None)