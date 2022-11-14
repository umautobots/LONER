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


PROJECT_ROOT = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 
                    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.cloner_slam import ClonerSLAM
from src.common.sensors import Image, LidarScan
from src.visualization.draw_frames import FrameDrawer

ROSBAG_PATH = os.path.expanduser("~/data/cloner_carla_dataset.bag")
LIDAR_TOPIC = "/carla/ego_vehicle/lidar/center"
IMAGE_TOPIC = "/carla/ego_vehicle/camera/rgb/image"

bridge = CvBridge()

def BuildScanFromMsg(lidar_msg, timestamp) -> LidarScan:
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg).transpose()
    
    xyz = torch.from_numpy(xyz).float()

    dists = torch.linalg.norm(xyz, axis=0)
    directions = xyz / dists
    timestamps = torch.tile(torch.Tensor([timestamp.to_sec()]), (directions.shape[1],))
    return LidarScan(directions, dists, torch.eye(4), timestamps)

def BuildImageFromMsg(image_msg, timestamp) -> Image:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    pytorch_img = torch.from_numpy(cv_img)
    return Image(pytorch_img, timestamp.to_sec())

if __name__ == "__main__":
    rospy.init_node('Cloner-SLAM')

    start_time = None

    print("Opening Rosbag")
    bag = rosbag.Bag(ROSBAG_PATH, 'r')
    print("Rosbag Open")

    cloner_slam = ClonerSLAM("../cfg/default_settings.yaml")

    drawer = FrameDrawer(cloner_slam._frame_signal)

    cloner_slam.Start()

    init = False
    prev_time = None
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
            cloner_slam.ProcessRGB(image)
        elif topic == LIDAR_TOPIC:
            lidar_scan = BuildScanFromMsg(msg, t)
            cloner_slam.ProcessLidar(lidar_scan)
        else:
            raise RuntimeError("Should be unreachable.")

        # if t.to_sec() - start_time > 5:
        #     break
        # print("sleeping", t.to_sec() - prev_time)
        
        drawer.Update()

        time.sleep(t.to_sec() - prev_time)
        prev_time = t.to_sec()
        
    bag.close()
    cloner_slam.Stop()