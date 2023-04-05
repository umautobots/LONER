"""
Uses the groundtruth trajectory and measured lidar scans to reconstruct a lidar map.
"""



from pathlib import Path
import numpy as np 
import argparse
import rosbag
import tqdm
import ros_numpy
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from utils import *
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser("Create GT Map")
parser.add_argument("rosbag_path", type=str)
parser.add_argument("--voxel_size", type=float, default=0.05)
parser.add_argument("--groundtruth_traj", type=str, default=None)
parser.add_argument("--run_outlier_filter", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="reconstructed_map.pcd")
args = parser.parse_args()

bag = rosbag.Bag(args.rosbag_path)

LIDAR_MIN_RANGE = 0.5 #https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p0-os1.pdf

# Get ground truth trajectory
rosbag_path = Path(args.rosbag_path)

if args.groundtruth_traj is None:
    ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
else:
    ground_truth_file = args.groundtruth_traj
ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")

ground_truth_data = ground_truth_df.to_numpy(dtype=np.float128)

gt_timestamps = ground_truth_data[:,0]
xyz = ground_truth_data[:,1:4]
quats = ground_truth_data[:,4:]

rotations = Rotation.from_quat(quats)
slerp = Slerp(gt_timestamps, rotations)
xyz_interp = interp1d(gt_timestamps, xyz, axis=0)


def process_cloud(bag_it):
    _,msg,timestamp = bag_it

    if timestamp.to_sec() < gt_timestamps[0] or timestamp.to_sec() > gt_timestamps[-1]:
        return None

    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

    num_points = msg.width * msg.height
    xyz = np.zeros((num_points, 3,), dtype=np.float64)
    xyz[:,0] = lidar_data['x'].reshape(-1,)
    xyz[:,1] = lidar_data['y'].reshape(-1,)
    xyz[:,2] = lidar_data['z'].reshape(-1,)


    dists = np.linalg.norm(xyz, axis=1)
    
    valid_ranges = dists > LIDAR_MIN_RANGE
    xyz = xyz[valid_ranges]

    xyz_homog = np.hstack((xyz, np.ones_like(xyz[:,0:1]))).reshape(-1, 4, 1)
    

    fields = [f.name for f in msg.fields]

    time_key = None
    for f in fields:
        if "time" in f or f == "t":
            time_key = f
            break

    timestamps = lidar_data[time_key].reshape(-1,).astype(np.float128)[valid_ranges]

    if timestamps[-1] > 1e7:
        timestamps *= 1e-9
    if timestamps[0] < 1e-2:
        timestamps += timestamp.to_sec()
    else:
        timestamps = timestamps - timestamps[0] + timestamp.to_sec()

    if timestamps.max() > gt_timestamps[-1]:
        return None
    
    rotations = slerp(timestamps).as_matrix()
    translations = xyz_interp(timestamps).reshape(-1, 3, 1)
    T = np.concatenate((rotations, translations), axis=2)
    T = np.hstack((T, np.tile([0,0,0,1], (translations.shape[0],1,1))))

    compensated_points = T @ xyz_homog
    compensated_points = compensated_points.squeeze(2)[:,:3]

    output_pcd = o3d.cuda.pybind.geometry.PointCloud()
    output_pcd.points.extend(compensated_points)

    output_pcd = output_pcd.voxel_down_sample(voxel_size=args.voxel_size)

    return np.asarray(output_pcd.points)


clouds = process_map(process_cloud, bag.read_messages(topics=["/os_cloud_node/points"]), max_workers=30, total=bag.get_message_count("/os_cloud_node/points"))

result_pcd = o3d.cuda.pybind.geometry.PointCloud()
for cloud in tqdm.tqdm(clouds):
    if cloud is None:
        continue
    result_pcd.points.extend(o3d.cuda.pybind.utility.Vector3dVector(cloud))

print("Downsampling")
result_pcd = result_pcd.voxel_down_sample(voxel_size=args.voxel_size)

if args.run_outlier_filter:
    print("Running SOR filter")
    _, ind = result_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    result_pcd = result_pcd.select_by_index(ind)

o3d.io.write_point_cloud(args.output_path, result_pcd)
    
    