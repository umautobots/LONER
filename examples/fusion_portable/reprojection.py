from pathlib import Path
import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
import rosbag
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import tf2_msgs.msg
import tqdm
import rospy
import ros_numpy
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from utils import *
from more_itertools import peekable

LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf
MOTION_COMPENSATE = False
IM_COMPRESSED = True

parser = argparse.ArgumentParser("stereo to rgbd")
parser.add_argument("rosbag_path", type=str)
parser.add_argument("calib_path", type=str)

args = parser.parse_args()

bridge = CvBridge()

### Load in the calibration
calibration = FusionPortableCalibration(args.calib_path)
K_left = calibration.left_cam_intrinsic["K"]
proj_left = calibration.left_cam_intrinsic["projection_matrix"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
rect_left = calibration.left_cam_intrinsic["rectification_matrix"]
im_width, im_height = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])
K_right = calibration.right_cam_intrinsic["K"]
proj_right = calibration.right_cam_intrinsic["projection_matrix"]
distortion_right = calibration.right_cam_intrinsic["distortion_coeffs"]
rect_right = calibration.right_cam_intrinsic["rectification_matrix"]
xmap_left, ymap_left = cv2.initUndistortRectifyMap(K_left, distortion_left, rect_left, proj_left, (im_width, im_height), cv2.CV_32FC1)
xmap_right, ymap_right = cv2.initUndistortRectifyMap(K_right, distortion_right, rect_right, proj_right, (im_width, im_height), cv2.CV_32FC1)

lidar_to_left_cam = calibration.t_lidar_to_left_cam
xyz = lidar_to_left_cam["xyz"]
xyz = np.array(xyz).reshape(3,1)
rotation = lidar_to_left_cam["orientation"]
rotation[0], rotation[3] = rotation[3], rotation[0]
rotmat = Rotation.from_quat(rotation).as_matrix()
lidar_to_left_cam = np.hstack((rotmat, xyz))
lidar_to_left_cam = np.vstack((lidar_to_left_cam, [0,0,0,1]))  

### Load GT poses
rosbag_path = Path(args.rosbag_path)

if MOTION_COMPENSATE:
    ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
    ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    ground_truth_data = ground_truth_df.to_numpy(dtype=np.float128)
    gt_timestamps = ground_truth_data[:,0]
    xyz = ground_truth_data[:,1:4]
    quats = ground_truth_data[:,4:]
    rotations = Rotation.from_quat(quats)
    pose_rot_slerp = Slerp(gt_timestamps, rotations)
    pose_xyz_interp = interp1d(gt_timestamps, xyz, axis=0)

compressed_str = "/compressed" if IM_COMPRESSED else ""
cam_topics = [f"/stereo/frame_left/image_raw{compressed_str}", f"/stereo/frame_right/image_raw{compressed_str}"]
all_topics = cam_topics + ["/os_cloud_node/points"]

### Go through bag and get images and corresponding nearby lidar scans
bag = rosbag.Bag(rosbag_path.as_posix())
bag_it = peekable(bag.read_messages(topics=all_topics))

for idx in tqdm.trange(bag.get_message_count(cam_topics)//2):
    try:
        # Get left and right image, plus a lidar scan. Imperfect time sync is handled with global motion compensation
        lidar_msg = None
        while bag_it.peek()[0] == "/os_cloud_node/points":
            lidar_topic, lidar_msg, lidar_ts = next(bag_it)
            lidar_msg.header.frame_id = "lidar"
        
        topic1, msg1, timestamp1 = next(bag_it)
        
        while bag_it.peek()[0] == "/os_cloud_node/points":
            lidar_topic, lidar_msg, lidar_ts = next(bag_it)
            lidar_msg.header.frame_id = "lidar"
        
        _, msg2, timestamp2 = next(bag_it)

        while bag_it.peek()[0] == "/os_cloud_node/points":
            lidar_topic, lidar_msg, lidar_ts = next(bag_it)
            lidar_msg.header.frame_id = "lidar"
    except StopIteration:
        break


    # Make sure the image pair is actually synchronized
    if (timestamp2 - timestamp1).to_sec() > 0.02:
        print("Timestamps too far apart")
        continue

    if "left" in topic1:
        left_msg = msg1
        right_msg = msg2
        left_ts = timestamp1
    else:
        right_msg = msg1
        left_msg = msg2
        left_ts = timestamp2

    if IM_COMPRESSED:
        left_im = bridge.compressed_imgmsg_to_cv2(left_msg, 'bgr8')
        right_im = bridge.compressed_imgmsg_to_cv2(right_msg, 'bgr8')
    else:
        left_im = bridge.imgmsg_to_cv2(left_msg, 'bgr8')
        right_im = bridge.imgmsg_to_cv2(right_msg, 'bgr8')

    # Only keep images/scans when we (a) have a lidar scan and (b) all times are within the GT pose time range
    if MOTION_COMPENSATE and (left_ts.to_sec() < gt_timestamps[0] or left_ts.to_sec() + 0.1 > gt_timestamps[-1]):
        continue
    if lidar_msg is None or (MOTION_COMPENSATE and (lidar_ts.to_sec() < gt_timestamps[0] or lidar_ts.to_sec() + 0.1 > gt_timestamps[-1])):
        continue

    print("Lidar TS, Camera TS", left_ts.to_sec(), lidar_ts.to_sec())
    # Undistort the images
    left_im = cv2.undistort(left_im, K_left, distortion_left)
    right_im = cv2.undistort(right_im, K_right, distortion_right)

    # left_im = cv2.remap(left_im,
    #             xmap_left,
    #             ymap_left,
    #             cv2.INTER_LANCZOS4,
    #             cv2.BORDER_CONSTANT,
    #             0)
    # right_im = cv2.remap(right_im,
    #             xmap_right,
    #             ymap_right,
    #             cv2.INTER_LANCZOS4,
    #             cv2.BORDER_CONSTANT,
    #             0)

    # Get lidar points
    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
    lidar_data = pd.concat(list(map(pd.DataFrame, lidar_data))).to_numpy()
    xyz = lidar_data[:, :3]
    dists = np.linalg.norm(xyz, axis=1)
    valid_ranges = dists > LIDAR_MIN_RANGE
    xyz = xyz[valid_ranges]

    xyz_homog = np.hstack((xyz, np.ones_like(xyz[:,0:1]))).reshape(-1, 4, 1)
    lidar_point_timestamps = lidar_data[valid_ranges, -1].astype(np.float128) + lidar_ts.to_sec()
    
    # Apply motion compensation (ends up being nothing if MOTION_COMPENSATE is false)
    if MOTION_COMPENSATE:
        lidar_pose_rotations = pose_rot_slerp(lidar_point_timestamps).as_matrix()
        lidar_pose_translations = pose_xyz_interp(lidar_point_timestamps).reshape(-1, 3, 1)
        lidar_poses = np.concatenate((lidar_pose_rotations, lidar_pose_translations), axis=2)
        lidar_poses = np.hstack((lidar_poses, np.tile([0,0,0,1], (lidar_pose_translations.shape[0],1,1)))).astype(np.float64)
        motion_compensated_points = lidar_poses @ xyz_homog
        motion_compensated_points = motion_compensated_points.squeeze(2)[:,:3]
    else:
        motion_compensated_points = xyz

    # Get the pose of the lidar at the time the image was captured
    if MOTION_COMPENSATE:
        lidar_pose_rot_cam_time = pose_rot_slerp(left_ts.to_sec()).as_matrix()
        lidar_pose_xyz_cam_time = pose_xyz_interp(left_ts.to_sec()).reshape(3, 1)
        lidar_pose_cam_time = np.hstack((lidar_pose_rot_cam_time, lidar_pose_xyz_cam_time))
        lidar_pose_cam_time = np.vstack((lidar_pose_cam_time, [0,0,0,1])).astype(np.float64)
    else:
        lidar_pose_cam_time = np.eye(4)

    world_to_camera = lidar_pose_cam_time @ lidar_to_left_cam

    # Project points into camera frame
    motion_compensated_points_homog = cv2.convertPointsToHomogeneous(motion_compensated_points).squeeze(1).T
    # This is actually lidar_to_camera when MOTION_COMPENSATE is false
    camera_frame_points = (np.linalg.inv(world_to_camera) @ motion_compensated_points_homog)[:3].T
    valid_points = camera_frame_points[:,2] > 0
    camera_frame_points = camera_frame_points[valid_points]
    # image_frame_points = cv2.projectPoints(camera_frame_points, np.zeros(3), np.zeros(3), proj_left[:3,:3], distortion_left)[0].squeeze(1).astype(int)

    # H = np.linalg.inv(world_to_camera)
    # motion_compensated_points = motion_compensated_points[:,]
    # image_frame_points = cv2.projectPoints(motion_compensated_points[valid_points].T, cv2.Rodrigues(H[:3,:3])[0], H[:3,3], K_left, distortion_left)[0].squeeze(1).astype(int)

    # Project into image frame
    image_frame_points_homog = proj_left[:3,:3] @ camera_frame_points.T
    image_frame_points = cv2.convertPointsFromHomogeneous(image_frame_points_homog.T).squeeze(1).astype(int)

    # Check in camera frame
    good_points = image_frame_points[:,0] >= 0
    good_points = np.logical_and(good_points, image_frame_points[:,0] < im_width)
    good_points = np.logical_and(good_points, image_frame_points[:,1] >= 0)
    good_points = np.logical_and(good_points, image_frame_points[:,1] < im_height)

    # Extract depths
    image_frame_points = image_frame_points[good_points]
    camera_frame_points = camera_frame_points[good_points]
    depths = camera_frame_points[...,2]


    # Colorize and draw
    depths_viz = cv2.normalize(depths, depths.copy(), alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    depths_viz = cv2.applyColorMap(depths_viz.astype(np.uint8), cv2.COLORMAP_TURBO).squeeze(1)

    disp_im = left_im.copy()
    
    # Expand the number of pixels we draw
    depth_im = np.zeros_like(disp_im)
    depth_im[image_frame_points[:,1],image_frame_points[:,0]] = depths_viz
    depth_im = cv2.dilate(depth_im, np.ones(3))
    color_coords = depth_im[:,:] != np.zeros(3)
    disp_im[color_coords] = depth_im[color_coords]


    cv2.imshow("Projected Points", disp_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()