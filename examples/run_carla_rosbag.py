#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import ros_numpy
import rosbag
import rospy
import tf2_py
import torch
from attrdict import AttrDict
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.cloner_slam import ClonerSLAM
from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.sensors import Image, LidarScan
from src.common.settings import Settings
from src.logging.default_logger import DefaultLogger
from src.logging.draw_frames_to_ros import FrameDrawer

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


def build_poses_from_buffer(tf_buffer, timestamps, camera, lidar):
    camera_poses = []
    lidar_poses = []
    for t in timestamps:
        try:
            camera_tf = tf_buffer.lookup_transform_core('map', camera, t)
            lidar_tf = tf_buffer.lookup_transform_core('map', lidar, t)
            camera_poses.append(msg_to_transformation_mat(camera_tf))
            lidar_poses.append(msg_to_transformation_mat(lidar_tf))
        except Exception as e:
            print("Skipping invalid tf")

    return torch.stack(camera_poses).float(), torch.stack(lidar_poses).float()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run ClonerSLAM on RosBag")
    parser.add_argument("rosbag_path")
    args = parser.parse_args()

    start_time = None

    bag = rosbag.Bag(args.rosbag_path, 'r')

    settings = Settings.load_from_file("../cfg/default_settings.yaml")

    init = False
    prev_time = None

    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))
    # Build the TF Buffer ahead of time so we can interpolate instead of extrapolate
    start_time = None
    end_time = None

    ego_pose_timestamps = []

    camera = settings.ros_names.camera
    lidar = settings.ros_names.lidar
    camera_suffix = settings.ros_names.camera_suffix
    topic_prefix = settings.ros_names.topic_prefix

    camera_info_topic = f"{topic_prefix}/{camera}/camera_info" 
    image_topic = f"{topic_prefix}/{camera}/{camera_suffix}"
    lidar_topic = f"{topic_prefix}/{lidar}"

    found_intrinsic = False
    for topic, msg, t in bag.read_messages(topics=["/tf", camera_info_topic]):
        if topic == "/tf":
            for tf_msg in msg.transforms:
                tf_buffer.set_transform(tf_msg, "default_authority")

                if tf_msg.header.frame_id == "map" and tf_msg.child_frame_id == "ego_vehicle":
                    ego_pose_timestamps.append(t)

            if start_time is None:
                start_time = t
            end_time = t
        elif topic == camera_info_topic and not found_intrinsic:
            found_intrinsic = True

            k_list = list(msg.K)
            settings["calibration"]["camera_intrinsic"]["k"] = torch.Tensor(
                k_list).reshape(3, 3)
            settings["calibration"]["camera_intrinsic"]["height"] = msg.height
            settings["calibration"]["camera_intrinsic"]["width"] = msg.width
            settings["calibration"]["camera_intrinsic"]["distortion"] = torch.Tensor(msg.D)
    
    
    bag.close()

    # Build the camera poses and lidar poses
    cam_poses, lidar_poses = build_poses_from_buffer(
        tf_buffer, ego_pose_timestamps, camera, lidar)

    t_avg = rospy.Time.from_sec((start_time.to_sec() + end_time.to_sec())/2)

    lidar_to_camera = tf_to_settings(
        tf_buffer.lookup_transform_core(lidar, camera, t_avg))

    settings["calibration"]["lidar_to_camera"] = lidar_to_camera

    cloner_slam = ClonerSLAM(settings)
    ray_range = settings.mapper.optimizer.model_config.data.ray_range
    image_size = (settings.calibration.camera_intrinsic.height,
                  settings.calibration.camera_intrinsic.width)

    camera_to_lidar = Pose.from_settings(lidar_to_camera, True).get_transformation_matrix().inverse()

    cloner_slam.initialize(camera_to_lidar, lidar_poses,  settings.calibration.camera_intrinsic.k,
                           ray_range, image_size, args.rosbag_path)

    if PUB_ROS:
        rospy.init_node('cloner_slam')
        drawer = FrameDrawer(cloner_slam._frame_signal,
                             cloner_slam._rgb_signal,
                             cloner_slam.get_world_cube())
    else:
        drawer = DefaultLogger(cloner_slam._frame_signal,
                                cloner_slam._keyframe_update_signal,
                                cloner_slam.get_world_cube(),
                                settings.calibration)

    for f in glob.glob("../outputs/frame*"):
        shutil.rmtree(f)

    # TODO: Prevent duplicate opens
    bag = rosbag.Bag(args.rosbag_path, 'r')

    cloner_slam.start()

    start_time = None
    for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic]):
        
        if start_time is None:
            start_time = t.to_sec()

        # if t.to_sec() - start_time > 2:
        #     break

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
