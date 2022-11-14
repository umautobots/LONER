import rospy

import numpy as np
from common.signals import Slot, Signal
from common.frame import Frame
from scipy.spatial.transform import Rotation as R, Slerp

import tf2_ros
import geometry_msgs.msg

class FrameDrawer:
    def __init__(self, frame_signal: Signal):
        self._frame_slot = frame_signal.Register()
        self._broadcater = tf2_ros.TransformBroadcaster()

    def Update(self):
        while self._frame_slot.HasValue():
            frame = self._frame_slot.GetValue()
            start_tf = frame.GetStartLidarPose().GetTransformationMatrix().detach().cpu().numpy()

            translation = start_tf[:3, 3]
            rotation = R.from_matrix(start_tf[:3, :3]).as_quat()
            
            transform_msg = geometry_msgs.msg.TransformStamped()
            transform_msg.header.stamp = rospy.Time.now()
            transform_msg.header.frame_id = "world"
            transform_msg.child_frame_id = "lidar_start_pose"
            transform_msg.transform.translation.x = translation[0]
            transform_msg.transform.translation.y = translation[1]
            transform_msg.transform.translation.z = translation[2]
            
            transform_msg.transform.rotation.x = rotation[0]
            transform_msg.transform.rotation.y = rotation[1]
            transform_msg.transform.rotation.z = rotation[2]
            transform_msg.transform.rotation.w = rotation[3]

            self._broadcater.sendTransform(transform_msg)
            
