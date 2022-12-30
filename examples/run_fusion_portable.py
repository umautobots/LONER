#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import sys

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
from sensor_msgs.msg import Image

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from fusion_portable_calibration import FusionPortableCalibration

from src.cloner_slam import ClonerSLAM
from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.sensors import Image, LidarScan
from src.common.settings import Settings
from src.visualization.draw_frames_to_mpl import MplFrameDrawer
from src.visualization.draw_frames_to_ros import FrameDrawer

# autopep8: on


bridge = CvBridge()

def build_scan_from_msg(lidar_msg, timestamp) -> LidarScan:
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
        lidar_msg).transpose()

    xyz = torch.from_numpy(xyz).float()

    dists = torch.linalg.norm(xyz, axis=0)
    directions = xyz / dists
    timestamps = torch.tile(torch.Tensor(
        [timestamp.to_sec()]), (directions.shape[1],))
    return LidarScan(directions, dists, torch.eye(4), timestamps)


def tf_to_settings(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    xyz = [trans.x, trans.y, trans.z]
    quat = [rot.x, rot.y, rot.z, rot.w]

    return AttrDict({"xyz": xyz, "orientation": quat})


def build_image_from_msg(image_msg, timestamp) -> Image:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    pytorch_img = torch.from_numpy(cv_img / 255)
    return Image(pytorch_img, timestamp.to_sec())


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3, 1)
    rotmat = torch.from_numpy(R.from_quat(
        [rot.x, rot.y, rot.z, rot.w]).as_matrix())

    T = torch.hstack((rotmat, xyz))
    T = torch.vstack((T, torch.Tensor([0, 0, 0, 1])))
    return T

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

def build_poses_from_buffer(tf_buffer, timestamps):
    lidar_poses = []
    for t in timestamps:
        try:
            lidar_tf = tf_buffer.lookup_transform_core('map', "lidar", t)
            lidar_poses.append(msg_to_transformation_mat(lidar_tf))
        except Exception as e:
            print("Skipping invalid tf")

    return torch.stack(lidar_poses).float()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run ClonerSLAM on RosBag")
    parser.add_argument("rosbag_path")
    parser.add_argument("calibration_path")
    args = parser.parse_args()

    calibration = FusionPortableCalibration(args.calibration_path)

    settings = Settings.load_from_file("../cfg/default_settings.yaml")

    init = False
    prev_time = None

    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))


    ego_pose_timestamps = []

    camera = settings.ros_names.camera
    lidar = settings.ros_names.lidar
    camera_suffix = settings.ros_names.camera_suffix
    topic_prefix = settings.ros_names.topic_prefix

    camera_info_topic = f"{topic_prefix}/{camera}/camera_info" 
    image_topic = f"{topic_prefix}/{camera}/{camera_suffix}"
    lidar_topic = f"{topic_prefix}/{lidar}"


    settings["calibration"]["lidar_to_camera"] = calibration.t_lidar_to_left_cam
    settings["calibration"]["camera_intrinsic"]["k"] = \
        torch.from_numpy(calibration.left_cam_intrinsic["K"])

    settings["calibration"]["camera_intrinsic"]["distortion"] = \
        torch.from_numpy(calibration.left_cam_intrinsic["distortion_coeffs"])

    settings["calibration"]["camera_intrinsic"]["width"] = calibration.left_cam_intrinsic["width"]
    
    settings["calibration"]["camera_intrinsic"]["height"] = calibration.left_cam_intrinsic["height"]

    cloner_slam = ClonerSLAM(settings)
    ray_range = settings.mapper.optimizer.model_config.data.ray_range
    image_size = (settings.calibration.camera_intrinsic.height,
                  settings.calibration.camera_intrinsic.width)

    camera_to_lidar = Pose.from_settings(calibration.t_lidar_to_left_cam, True).get_transformation_matrix().inverse()


    # Get ground truth trajectory
    rosbag_path = Path(args.rosbag_path)
    ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
    ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    tf_buffer, timestamps = build_buffer_from_df(ground_truth_df)
    lidar_poses = build_poses_from_buffer(tf_buffer, timestamps)

    cloner_slam.initialize(camera_to_lidar, lidar_poses, settings.calibration.camera_intrinsic.k,
                           ray_range, image_size, args.rosbag_path)

    if PUB_ROS:
        rospy.init_node('cloner_slam')
        drawer = FrameDrawer(cloner_slam._frame_signal,
                             cloner_slam._rgb_signal,
                             cloner_slam.get_world_cube())
    else:
        drawer = MplFrameDrawer(cloner_slam._frame_signal,
                                cloner_slam.get_world_cube())

    for f in glob.glob("../outputs/frame*"):
        shutil.rmtree(f)

    # TODO: Prevent duplicate opens
    bag = rosbag.Bag(args.rosbag_path, 'r')

    cloner_slam.start()

    start_time = None
    for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic]):
        
        if start_time is None:
            start_time = t.to_sec()

        if t.to_sec() - start_time > 10:
            break

        # Wait for lidar to init
        if topic == lidar_topic and not init:
            init = True
            start_time = t.to_sec()
            prev_time = start_time
        if not init:
            continue

        if topic == image_topic:
            image = build_image_from_msg(msg, t)

            try:
                camera_tf = tf_buffer.lookup_transform_core('map', camera, t)
            except tf2_py.ExtrapolationException as e:
                continue

            T = msg_to_transformation_mat(camera_tf)

            camera_pose = Pose(T)
            cloner_slam.process_rgb(image, camera_pose)
        elif topic == lidar_topic:
            lidar_scan = build_scan_from_msg(msg, t)
            cloner_slam.process_lidar(lidar_scan)
        else:
            raise Exception("Should be unreachable")

    cloner_slam.stop(drawer.update, drawer.finish)
