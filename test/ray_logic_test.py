import os
import sys
import torch
import unittest
import yaml
import rosbag
import rospy
import numpy as np
import open3d as o3d


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.settings import Settings
from src.common.sensors import LidarScan, Image
from src.common.frame import Frame
from src.mapping.keyframe import KeyFrame
from src.common.ray_utils import rays_to_pcd, rays_to_o3d

import argparse
import os
import sys

import cv2
import pandas as pd
import ros_numpy
import rosbag
import rospy
import tf2_py
import torch
from attrdict import AttrDict
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from examples.fusion_portable_calibration import FusionPortableCalibration


LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf
IM_SCALE_FACTOR = 0.5
LIDAR_TOPIC = "/os_cloud_node/points"

bridge = CvBridge()

def build_scan_from_msg(lidar_msg: PointCloud2, timestamp: rospy.Time) -> LidarScan:
    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(
        lidar_msg)

    lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
    xyz = lidar_data[:, :3]
    
    dists = torch.linalg.norm(xyz, dim=1)
    valid_ranges = dists > LIDAR_MIN_RANGE


    xyz = xyz[valid_ranges].T
    timestamps = (lidar_data[valid_ranges, -1] + timestamp.to_sec()).float()
    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:, indices]

    return LidarScan(directions.float(), dists.float(), torch.eye(4).float(), timestamps.float())

def build_o3d_from_msg(lidar_msg: PointCloud2):
    pcd = o3d.cuda.pybind.geometry.PointCloud()

    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(
        lidar_msg)

    lidar_data = pd.DataFrame(lidar_data).to_numpy()
    
    xyz = lidar_data[:, :3]

    dists = np.linalg.norm(xyz, axis=1)
    valid_ranges = dists > LIDAR_MIN_RANGE
    xyz = xyz[valid_ranges]
    
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def build_buffer_from_df(df: pd.DataFrame):
    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))
    timestamps = []
    for _, row in df.iterrows():
        ts = rospy.Time.from_sec(row["timestamp"])
        timestamps.append(ts)

        new_transform = TransformStamped()
        new_transform.header.frame_id = "map"
        new_transform.header.stamp = ts
        new_transform.child_frame_id = "lidar"

        new_transform.transform.translation.x = row["x"]
        new_transform.transform.translation.y = row["y"]
        new_transform.transform.translation.z = row["z"]

        new_transform.transform.rotation.x = row["q_x"]
        new_transform.transform.rotation.y = row["q_y"]
        new_transform.transform.rotation.z = row["q_z"]
        new_transform.transform.rotation.w = row["q_w"]

        tf_buffer.set_transform(new_transform, "default_authority")

    return tf_buffer, timestamps


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3, 1)
    rotmat = torch.from_numpy(R.from_quat(
        [rot.x, rot.y, rot.z, rot.w]).as_matrix())

    T = torch.hstack((rotmat, xyz))
    T = torch.vstack((T, torch.Tensor([0, 0, 0, 1])))
    return T.float()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test Ray Logic")
    parser.add_argument("rosbag_path")
    parser.add_argument("calibration_path")
    args = parser.parse_args()

    calibration = FusionPortableCalibration(args.calibration_path)
    lidar_to_camera = Pose.from_settings(calibration.t_lidar_to_left_cam)
    
    settings = Settings.load_from_file("../cfg/default_settings.yaml")

    # Get ground truth trajectory
    rosbag_path = Path(args.rosbag_path)
    ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
    ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    tf_buffer, timestamps = build_buffer_from_df(ground_truth_df)

    bag = rosbag.Bag(args.rosbag_path, 'r')

    lidar_scans = []
    o3d_clouds = []

    init_time = None
    start_time = None
    end_time = None
    start_cloud = None
    end_cloud = None
    start_scan = None
    end_scan = None
    for topic, msg, timestamp in bag.read_messages(topics=[LIDAR_TOPIC]):        
        
        if timestamp < timestamps[0]:
            continue

        if init_time is None:
            init_time = timestamp

        timestamp -= init_time
        
        if timestamp.to_sec() < 2:
            continue
        elif timestamp.to_sec() < 60 and start_time is not None:
            continue
        
        if start_cloud is None:
            start_time = timestamp
            start_cloud = build_o3d_from_msg(msg)
            start_scan = build_scan_from_msg(msg, timestamp)
        elif end_cloud is None:
            end_time = timestamp
            end_cloud = build_o3d_from_msg(msg)
            end_scan = build_scan_from_msg(msg, timestamp)
            break


    start_start_lidar_pose = tf_buffer.lookup_transform_core('map', "lidar", init_time + rospy.Duration.from_sec(start_scan.timestamps[0]))
    start_start_lidar_pose = msg_to_transformation_mat(start_start_lidar_pose).numpy()

    start_end_lidar_pose = tf_buffer.lookup_transform_core('map', "lidar", init_time + rospy.Duration.from_sec(start_scan.timestamps[-1]))
    start_end_lidar_pose = msg_to_transformation_mat(start_end_lidar_pose).numpy()


    end_start_lidar_pose = tf_buffer.lookup_transform_core('map', 'lidar', init_time + rospy.Duration.from_sec(end_scan.timestamps[0]))
    end_start_lidar_pose = msg_to_transformation_mat(end_start_lidar_pose).numpy()

    end_end_lidar_pose = tf_buffer.lookup_transform_core('map', 'lidar', init_time + rospy.Duration.from_sec(end_scan.timestamps[-1]))
    end_end_lidar_pose = msg_to_transformation_mat(end_end_lidar_pose).numpy()
    
    start_cloud.transform(start_start_lidar_pose)
    start_cloud.paint_uniform_color([1, 0.706, 0])
    end_cloud.transform(end_start_lidar_pose)
    end_cloud.paint_uniform_color([0, 0.651, 0.929])


    start_frame = Frame(None, None, start_scan, lidar_to_camera)
    start_frame._lidar_start_pose = Pose(torch.from_numpy(start_start_lidar_pose))
    start_frame._lidar_end_pose = Pose(torch.from_numpy(start_end_lidar_pose))
    start_frame.start_image = Image(torch.zeros(1), start_time.to_sec())
    start_frame.end_image = Image(torch.zeros(1), start_time.to_sec() + start_scan.timestamps[-1].item())
    start_kf = KeyFrame(start_frame, 'cpu')

    end_frame = Frame(None, None, end_scan, lidar_to_camera)
    end_frame._lidar_start_pose = Pose(torch.from_numpy(end_start_lidar_pose))
    end_frame._lidar_end_pose = Pose(torch.from_numpy(end_end_lidar_pose))
    end_frame.start_image = Image(torch.zeros(1), end_time.to_sec())
    end_frame.end_image = Image(torch.zeros(1), end_time.to_sec() + end_scan.timestamps[-1].item())
    end_kf = KeyFrame(end_frame, 'cpu')

    ray_range = torch.Tensor(settings.mapper.optimizer.model_config.model.ray_range)
    world_cube = WorldCube(torch.tensor([1]), torch.tensor([0,0,0]))

    lidar_indices = torch.arange(0, len(start_scan), 10)
    start_rays, start_depths = start_kf.build_lidar_rays(lidar_indices, ray_range, world_cube, False, True)
    
    start_o3d_from_rays = rays_to_o3d(start_rays, start_depths).paint_uniform_color([1,0,0])

    end_rays, end_depths = end_kf.build_lidar_rays(lidar_indices, ray_range, world_cube, False, True)
    end_o3d_from_rays = rays_to_o3d(end_rays, end_depths).paint_uniform_color([0,0,1])


    o3d.visualization.draw_geometries([start_cloud, end_cloud, start_o3d_from_rays, end_o3d_from_rays])
