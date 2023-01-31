# coding: utf-8

import argparse
import os
import pickle
import sys
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import numpy as np
import rosbag
from geometry_msgs.msg import PoseStamped
import rospy

# Every <this many> meters, do a 360
SPIN_SPACING_M = 10

# how long in seconds should the 360 takes
SPIN_DURATION_S = 5

# How fast should the fly-through move, in m/s
CAMERA_VELOCITY = 1.5

FPS = 10

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
args = parser.parse_args()


with open(f"{args.output_dir}/full_config.pkl", 'rb') as f:
    settings: Settings = pickle.load(f)


# Get ground truth trajectory
rosbag_path = Path(settings.dataset_path)
ground_truth_file = "../../data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt"
ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")

ground_truth_data = ground_truth_df.to_numpy(dtype=np.float128)

gt_xyz = ground_truth_data[:,1:4]
gt_quats = ground_truth_data[:,4:]

diffs = np.diff(gt_xyz, axis=0)
dists = np.sqrt( np.sum(diffs ** 2, axis=1))

# Normalize the camera velocity
dts = dists / CAMERA_VELOCITY
timestamps = np.cumsum(dts)
timestamps = np.insert(timestamps, 0, 0.)

rotations = Rotation.from_quat(gt_quats)
slerp = Slerp(timestamps, rotations)
xyz_interp = interp1d(timestamps, gt_xyz, axis=0)

num_images = int(timestamps[-1] * FPS)
image_timestamps = np.linspace(0, timestamps[-1], num_images)

image_poses = []

dist_since_last_spin = 0
prev_pose = np.eye(4)
for timestamp in image_timestamps:

    xyz = xyz_interp(timestamp)
    rot = slerp(timestamp)

    T = np.hstack((rot.as_matrix(), xyz.reshape(-1, 1)))
    T = np.vstack((T, [0,0,0,1]))
    image_poses.append(T)
    
    dist_since_last_spin += np.sqrt(np.sum((xyz - prev_pose[:3,3])**2))

    print(dist_since_last_spin, SPIN_SPACING_M)

    if dist_since_last_spin > SPIN_SPACING_M:
        num_spin_steps =  SPIN_DURATION_S * FPS
        spin_amounts_rad = np.linspace(0, 2*np.pi, num_spin_steps)
        rotations = Rotation.from_euler('z', spin_amounts_rad)

        for rel_rot in rotations:
            T = np.hstack((rot.as_matrix() @ rel_rot.as_matrix(), xyz.reshape(-1, 1)))
            T = np.vstack((T, [0,0,0,1]))
            image_poses.append(T)

        dist_since_last_spin = 0
    prev_pose = T
bag = rosbag.Bag("test_poses.bag", 'w')



for pose, ts in zip(image_poses, image_timestamps):
    pose_msg = PoseStamped()

    pose_msg.header.stamp = rospy.Time.from_sec(ts)
    pose_msg.header.frame_id = "map"

    pose_msg.pose.position.x = pose[0,3]
    pose_msg.pose.position.y = pose[1,3]
    pose_msg.pose.position.z = pose[2,3]

    quat = Rotation.from_matrix(pose[:3,:3]).as_quat()

    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]

    bag.write("/camera_pose", pose_msg, pose_msg.header.stamp)

bag.close()