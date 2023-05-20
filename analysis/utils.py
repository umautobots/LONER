from sensor_msgs.msg import PointCloud2
import ros_numpy
import torch
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os

def lidar_ts_to_seq(bag, lidar_topic):
    init_ts = None
    lidar_ts_to_seq = []
    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
        if init_ts == None:
            init_ts = msg.header.stamp.to_sec()
        timestamp = msg.header.stamp.to_sec() - init_ts
        lidar_ts_to_seq.append(timestamp)
    return lidar_ts_to_seq

def first_lidar_ts(bag, lidar_topic, gt_timestamps):
    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
        if timestamp.to_sec() > gt_timestamps[0]:
            start_time = timestamp.to_sec()
            return start_time

def find_corresponding_lidar_scan(bag, lidar_topic, seq):
    for topic, msg, ts in bag.read_messages(topics=[lidar_topic]):
        if msg.header.seq == seq:
            return msg

def bag_to_lidar_msg_list(bag, lidar_topic):
    lidar_scan_list = []
    for topic, msg, ts in bag.read_messages(topics=[lidar_topic]):
        lidar_scan_list.append(msg)
    return lidar_scan_list

def o3d_pc_from_msg(lidar_msg: PointCloud2):
    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
    lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
    xyz = lidar_data[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

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

class TUMPose:
    def __init__(self, timestamp, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def to_transform(self):
        r = R.from_quat([self.qx, self.qy, self.qz, self.qw])
        r_mat = r.as_matrix()
        t = np.array([self.x, self.y, self.z])
        T = np.eye(4)
        T[:3,:3] = r_mat
        T[:3,3] = t
        return T

def find_pose_by_timestamp(poses, timestamp):
    for pose in poses:
        if pose.timestamp == timestamp:
            return pose
    raise ValueError(f"No pose found with timestamp {timestamp}")

def find_closest_pose_by_timestamp(poses, timestamp):
    closest_pose = None
    closest_ts = None
    closest_dist = float('inf')
    for pose in poses:
        dist = abs(pose.timestamp - timestamp)
        if dist < closest_dist:
            closest_pose = pose
            closest_dist = dist
            closest_ts = pose.timestamp
    return closest_pose

def load_tum_trajectory(filename):
    poses = []
    with open(os.path.expanduser(filename), 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = line.strip().split(' ')
            timestamp = float(values[0])
            x, y, z = [float(v) for v in values[1:4]]
            qx, qy, qz, qw = [float(v) for v in values[4:]]
            pose = TUMPose(timestamp, x, y, z, qx, qy, qz, qw)
            poses.append(pose)
    return poses

def tumposes_ts_to_list(tumposes):
    ts_to_index_ = []
    for tumpose in tumposes:
        ts_to_index_.append(tumpose.timestamp)
    return ts_to_index_

def ckptposes_ts_to_list(ckptposes):
    ts_to_index_ = []
    for ckptpose in ckptposes:
        ts_to_index_.append(ckptpose['timestamp'])
    return ts_to_index_

