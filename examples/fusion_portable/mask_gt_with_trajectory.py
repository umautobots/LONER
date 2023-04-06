"""
Creates versions of the GT mesh which is the intersection of the mesh and the lidar FOV
throughout the GT trajectory
"""

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
import pathlib


CHUNK_SIZE=512

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose import Pose
from src.common.sensors import LidarScan
from src.common.ray_utils import LidarRayDirections
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler
from src.common.pose_utils import WorldCube

DIST_THRESHOLD = 0.1 #10cm

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
    parser.add_argument("groundtruth_map_dir", type=str, help="folder with ground truth map")
    parser.add_argument("reconstructed_map", type=str, help="file created with create_lidar_map.py")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--merged_transform", type=float, nargs=16, help="if supplied, transforms the reconstructed map by this before masking.")
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}/scan", exist_ok=True)
    
    scan_names = os.listdir(f"{args.groundtruth_map_dir}/scan")
    scan_nums = [s[:-4] for s in scan_names]

    scan_poses = load_scan_poses(f"{args.groundtruth_map_dir}/transformation.yaml", scan_nums)

    reconstructed_map = o3d.io.read_point_cloud(args.reconstructed_map)

    if args.merged_transform is not None:
        tf = np.array(args.merged_transform).reshape(4,4)
        reconstructed_map.transform(tf)
        
    print("Masking merged scan")
    full_scan = o3d.io.read_point_cloud(f"{args.groundtruth_map_dir}/merged_scan.pcd")
    distances = full_scan.compute_point_cloud_distance(reconstructed_map)
    good_distances = np.asarray(distances) < DIST_THRESHOLD
    good_gt_scan_points = np.asarray(full_scan.points)[good_distances]
    full_scan.points = o3d.utility.Vector3dVector(good_gt_scan_points)
    o3d.io.write_point_cloud(f"{args.output_dir}/merged_scan.pcd", full_scan)

    # for scan in sorted(scan_nums):
    #     print("Masking scan", scan)
    #     T_world_lidar = scan_poses[scan]

    #     gt_scan = o3d.io.read_point_cloud(f"{args.groundtruth_map_dir}/scan/{scan}.pcd")

    #     gt_scan = gt_scan.transform(T_world_lidar) # put it in the world frame

    #     # dist from each point in GT to each point in reconstruction
    #     distances = gt_scan.compute_point_cloud_distance(reconstructed_map) 

    #     good_distances = np.asarray(distances) < DIST_THRESHOLD

    #     good_gt_scan_points = np.asarray(gt_scan.points)[good_distances]

    #     gt_scan.points = o3d.utility.Vector3dVector(good_gt_scan_points)

    #     gt_scan.transform(T_world_lidar.inverse()) # back to original frame

    #     o3d.io.write_point_cloud(f"{args.output_dir}/scan/{scan}.pcd", gt_scan)

    print("Wrote results to", args.output_dir)