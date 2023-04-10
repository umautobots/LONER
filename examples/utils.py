import rospy
import tf2_py
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
import torch
import pytorch3d.transforms

import geometry_msgs.msg
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)

from examples.fusion_portable.fusion_portable_calibration import FusionPortableCalibration


def build_buffer_from_poses(poses, gt_timestamps):
    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))
    timestamps = []

    for pose, ts in zip(poses, gt_timestamps):
        timestamps.append(rospy.Time.from_sec(ts))

        xyz = pose[:3, 3]
        quat = Rotation.from_matrix(pose[:3,:3]).as_quat()

        new_transform = geometry_msgs.msg.TransformStamped()
        new_transform.header.frame_id = "map"
        new_transform.header.stamp = timestamps[-1]
        new_transform.child_frame_id = "lidar"

        new_transform.transform.translation.x = xyz[0]
        new_transform.transform.translation.y = xyz[1]
        new_transform.transform.translation.z = xyz[2]

        new_transform.transform.rotation.x = quat[0]
        new_transform.transform.rotation.y = quat[1]
        new_transform.transform.rotation.z = quat[2]
        new_transform.transform.rotation.w = quat[3]

        tf_buffer.set_transform(new_transform, "default_authority")
    return tf_buffer, timestamps


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = np.array([trans.x, trans.y, trans.z]).reshape(3, 1)
    quat = np.array([rot.x, rot.y, rot.z, rot.w])
    rotmat: Rotation = Rotation.from_quat(quat)
    rotmat = rotmat.as_matrix()
    
    T = np.hstack((rotmat, xyz))
    T = np.vstack((T, [0,0,0,1]))
    return T

def transformation_mat_to_pose_msg(tf):
    pose_msg = geometry_msgs.msg.Pose()

    quat = Rotation.from_matrix(tf[:3,:3]).as_quat()
    trans = tf[:3,3]
    
    pose_msg.position.x = trans[0]
    pose_msg.position.y = trans[1]
    pose_msg.position.z = trans[2]

    pose_msg.orientation.x = quat[0]
    pose_msg.orientation.y = quat[1]
    pose_msg.orientation.z = quat[2]
    pose_msg.orientation.w = quat[3]

    return pose_msg

def transformation_mat_to_odom_msg(tf):
    odom_msg = Odometry()

    quat = Rotation.from_matrix(tf[:3,:3]).as_quat()
    trans = tf[:3,3]
    
    odom_msg.pose.pose.position.x = trans[0]
    odom_msg.pose.pose.position.y = trans[1]
    odom_msg.pose.pose.position.z = trans[2]

    odom_msg.pose.pose.orientation.x = quat[0]
    odom_msg.pose.pose.orientation.y = quat[1]
    odom_msg.pose.pose.orientation.z = quat[2]
    odom_msg.pose.pose.orientation.w = quat[3]

    return odom_msg


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3, 1)
    quat = torch.Tensor([rot.w, rot.x, rot.y, rot.z])
    rotmat = pytorch3d.transforms.quaternion_to_matrix(quat)
    
    T = torch.hstack((rotmat, xyz))
    T = torch.vstack((T, torch.Tensor([0, 0, 0, 1])))
    return T.float()

def load_calibration(dataset_family: str, calib_path: str):
    if dataset_family.lower() == "fusion_portable":
        return FusionPortableCalibration(calib_path)
    print("Warning: Supplied dataset has no calibration configured. Don't enable the camera!")
    return None