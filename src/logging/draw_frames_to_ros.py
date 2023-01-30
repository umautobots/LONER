import rospy

import numpy as np
from common.signals import Slot, Signal, StopSignal
from common.frame import Frame
from common.pose_utils import WorldCube
from scipy.spatial.transform import Rotation as R, Slerp
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tf2_ros
import geometry_msgs.msg

"""
Listens to data on the given signals, and publishes to ROS for visualization. 
"""


def transform_from_pose(pose, child_frame):
    tf_mat = pose.get_transformation_matrix().detach()

    translation = tf_mat[:3, 3]
    rotation = R.from_matrix(tf_mat[:3, :3]).as_quat()

    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = rospy.Time.now()
    transform_msg.header.frame_id = "world"
    transform_msg.child_frame_id = child_frame
    transform_msg.transform.translation.x = translation[0]
    transform_msg.transform.translation.y = translation[1]
    transform_msg.transform.translation.z = translation[2]

    transform_msg.transform.rotation.x = rotation[0]
    transform_msg.transform.rotation.y = rotation[1]
    transform_msg.transform.rotation.z = rotation[2]
    transform_msg.transform.rotation.w = rotation[3]

    pose_msg = geometry_msgs.msg.PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "world"
    pose_msg.pose.position.x = translation[0]
    pose_msg.pose.position.y = translation[1]
    pose_msg.pose.position.z = translation[2]

    pose_msg.pose.orientation.x = rotation[0]
    pose_msg.pose.orientation.y = rotation[1]
    pose_msg.pose.orientation.z = rotation[2]
    pose_msg.pose.orientation.w = rotation[3]

    return transform_msg, pose_msg


class FrameDrawer:
    def __init__(self, frame_signal: Signal, rgb_signal: Signal, world_cube: WorldCube):
        self._world_cube = world_cube
        self._frame_slot = frame_signal.register()
        self._rgb_slot = rgb_signal.register()
        self._broadcater = tf2_ros.TransformBroadcaster()

        self._path = Path()
        self._gt_path = Path()
        self._path.header.frame_id = "world"
        self._gt_path.header.frame_id = "world"

        self._path_pub = rospy.Publisher("/estimated_path", Path, queue_size=1)
        self._gt_path_pub = rospy.Publisher(
            "/ground_truth_path", Path, queue_size=1)
        self._frame_img_pub = rospy.Publisher(
            "/frame_image_raw", Image, queue_size=1)
        self._input_img_pub = rospy.Publisher(
            "/input_img", Image, queue_size=1)

        self._bridge = CvBridge()
        self._gt_pose_offset = None

    def update(self):
        while self._frame_slot.has_value():
            frame = self._frame_slot.get_value()

            if isinstance(frame, StopSignal):
                break

            if self._gt_pose_offset is None:
                start_pose = frame._gt_lidar_start_pose
                start_pose.transform_world_cube(self._world_cube, reverse=True)
                self._gt_pose_offset = start_pose.inv()

            lidar_pose = frame.get_start_lidar_pose()
            lidar_pose.transform_world_cube(self._world_cube, reverse=True)

            transform_msg, new_pose = transform_from_pose(
                lidar_pose, "lidar_start_pose")
            gt_transform_msg, gt_pose = transform_from_pose(
                self._gt_pose_offset*frame._gt_lidar_start_pose.transform_world_cube(self._world_cube, True), "gt_lidar_pose")

            self._broadcater.sendTransform(transform_msg)
            self._broadcater.sendTransform(gt_transform_msg)

            self._path.header.stamp = rospy.Time.now()
            self._path.poses.append(new_pose)
            self._path_pub.publish(self._path)

            self._gt_path.header.stamp = rospy.Time.now()
            self._gt_path.poses.append(gt_pose)
            self._gt_path_pub.publish(self._gt_path)

            img = self._bridge.cv2_to_imgmsg(
                frame.start_image.image.detach().cpu().numpy(), encoding="rgb8")

            self._frame_img_pub.publish(img)

        while self._rgb_slot.has_value():
            img, _ = self._rgb_slot.get_value()

            if isinstance(img, StopSignal):
                break

            img = self._bridge.cv2_to_imgmsg(
                img.image.cpu().numpy(), encoding='rgb8')
            self._input_img_pub.publish(img)
