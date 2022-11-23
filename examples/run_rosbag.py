#!/usr/bin/env python3

import rospy
import rosbag
from sensor_msgs.msg import Image, PointCloud2
import os
import sys
import torch
import cv2
from cv_bridge import CvBridge
import ros_numpy
import time
import glob
import shutil
import tf2_py
from scipy.spatial.transform import Rotation as R
import argparse
import yaml
from attrdict import AttrDict

PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 
                    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.cloner_slam import ClonerSLAM
from src.common.sensors import Image, LidarScan
from src.visualization.draw_frames_to_ros import FrameDrawer
from src.visualization.draw_frames_to_mpl import MplFrameDrawer
from src.common.pose_utils import Pose
from src.common.settings import Settings

# LIDAR = "ego_vehicle/lidar/center"
LIDAR = "ego_vehicle/lidar"
LIDAR_TOPIC = f"/carla/{LIDAR}"
CAMERA = "ego_vehicle/rgb_front"
# CAMERA = "ego_vehicle/camera/rgb"
IMAGE_TOPIC = f"/carla/{CAMERA}/image"

bridge = CvBridge()

def BuildScanFromMsg(lidar_msg, timestamp) -> LidarScan:
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg).transpose()

    xyz = torch.from_numpy(xyz).float()
    
    dists = torch.linalg.norm(xyz, axis=0)
    directions = xyz / dists
    timestamps = torch.tile(torch.Tensor([timestamp.to_sec()]), (directions.shape[1],))
    return LidarScan(directions, dists, torch.eye(4), timestamps)

def TfToSettings(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    xyz = [trans.x, trans.y, trans.z]
    quat = [rot.x, rot.y, rot.z, rot.w]

    return AttrDict({"xyz": xyz, "orientation": quat})

def BuildImageFromMsg(image_msg, timestamp) -> Image:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    pytorch_img = torch.from_numpy(cv_img)
    return Image(pytorch_img, timestamp.to_sec())

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run ClonerSLAM on RosBag")
    parser.add_argument("rosbag_path")
    args = parser.parse_args()


    start_time = None

    bag = rosbag.Bag(args.rosbag_path, 'r')

    with open("../cfg/default_settings.yaml") as settings_file:
        settings_yaml = yaml.load(settings_file, Loader=yaml.FullLoader)

    settings = Settings(settings_yaml)

    init = False
    prev_time = None

    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))
    # Build the TF Buffer ahead of time so we can interpolate instead of extrapolate
    start_time = None
    end_time = None
    for _, msg, t in bag.read_messages(topics=["/tf"]):
        for tf_msg in msg.transforms:
            tf_buffer.set_transform(tf_msg, "default_authorrity")
        if start_time is None:
            start_time = t
        end_time = t
    bag.close()

    t_avg = rospy.Time.from_sec((start_time.to_sec() + end_time.to_sec())/2)

    lidar_to_camera = TfToSettings(tf_buffer.lookup_transform_core(LIDAR, CAMERA, t_avg))

    settings["calibration"]["lidar_to_camera"] = lidar_to_camera

    device = settings["device"]

    cloner_slam = ClonerSLAM(settings)

    if PUB_ROS:
        rospy.init_node('cloner_slam')
        drawer = FrameDrawer(cloner_slam._frame_signal, cloner_slam._rgb_signal)
    else:
        drawer = MplFrameDrawer(cloner_slam._frame_signal)

    for f in glob.glob("../outputs/frame*"):
        shutil.rmtree(f)

    # TODO: Prevent duplicate opens    
    bag = rosbag.Bag(args.rosbag_path, 'r')

    cloner_slam.Start()

    for topic, msg, t in bag.read_messages(topics=[LIDAR_TOPIC, IMAGE_TOPIC]):
        # Wait for lidar to init
        if topic == LIDAR_TOPIC and not init:
            init = True
            start_time = t.to_sec()
            prev_time = start_time
        if not init:
            continue
            
        if topic == IMAGE_TOPIC:
            image = BuildImageFromMsg(msg, t)

            try:
                camera_tf = tf_buffer.lookup_transform_core('map', CAMERA, t)
            except tf2_py.ExtrapolationException as e:
                continue

            trans = camera_tf.transform.translation
            rot = camera_tf.transform.rotation
            xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3,1)
            rotmat = torch.from_numpy(R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix())

            T = torch.hstack((rotmat, xyz))
            T = torch.vstack((T, torch.Tensor([0,0,0,1])))

            camera_pose = Pose(T)
            cloner_slam.ProcessRGB(image, camera_pose)
        elif topic == LIDAR_TOPIC:
            lidar_scan = BuildScanFromMsg(msg, t)
            cloner_slam.ProcessLidar(lidar_scan)
        else:
            raise Exception("Should be unreachable")

    cloner_slam.Stop(drawer.Update, drawer.Finish)
    cloner_slam.Cleanup()
    del drawer