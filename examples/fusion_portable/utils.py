import pandas as pd
import rospy
import tf2_py
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TransformStamped, Pose


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
            lidar_poses.append(None)

    return lidar_poses


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
    pose_msg = Pose()

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