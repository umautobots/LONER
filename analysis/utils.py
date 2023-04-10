from sensor_msgs.msg import PointCloud2
import ros_numpy
import torch
import pandas as pd
import numpy as np
import open3d as o3d

def lidar_ts_to_seq(bag, lidar_topic):
    init_ts = None
    lidar_ts_to_seq = []
    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
        if init_ts == None:
            init_ts = msg.header.stamp.to_sec()
        timestamp = msg.header.stamp.to_sec() - init_ts
        lidar_ts_to_seq.append(timestamp)
    return lidar_ts_to_seq

def first_lidar_ts(bag, lidar_topic):
    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic]):
        init_ts = msg.header.stamp.to_sec()
        break
    return init_ts

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