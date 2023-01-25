import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
import rosbag
from cv_bridge import CvBridge
import tqdm
import ros_numpy
import pandas as pd
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
rect_left = calibration.left_cam_intrinsic["rectification_matrix"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
im_width, im_height = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])

lidar_to_left_cam = calibration.t_lidar_to_left_cam
xyz = lidar_to_left_cam["xyz"]
xyz = np.array(xyz).reshape(3,1)
rotation = lidar_to_left_cam["orientation"]
rotation[0], rotation[3] = rotation[3], rotation[0]
rotmat = Rotation.from_quat(rotation).as_matrix()
lidar_to_left_cam = np.hstack((rotmat, xyz))
lidar_to_left_cam = np.vstack((lidar_to_left_cam, [0,0,0,1]))  

# xmap_left, ymap_left = cv2.initUndistortRectifyMap(K_left, distortion_left, rect_left, proj_left, (im_width, im_height), cv2.CV_32FC1)

# Compute rectified extrinsics
# R_rect_cam = np.eye(4)
# R_rect_cam[0:3,0:3] = rect_left
# T_cam = np.eye(4)
# T_cam[0,3] = proj_left[0,3] / proj_left[0,0]
# left_cam_to_lidar_unrect = np.linalg.inv(lidar_to_left_cam)
# left_cam_to_lidar_rect = T_cam @ (R_rect_cam @ left_cam_to_lidar_unrect)
# lidar_to_left_cam = np.linalg.inv(left_cam_to_lidar_rect)

### Load GT poses
compressed_str = "/compressed" if IM_COMPRESSED else ""
cam_topics = [f"/stereo/frame_left/image_raw{compressed_str}", f"/stereo/frame_right/image_raw{compressed_str}"]
all_topics = cam_topics + ["/os_cloud_node/points"]

### Go through bag and get images and corresponding nearby lidar scans
bag = rosbag.Bag(args.rosbag_path)
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

    if lidar_msg is None:
        continue
    
    # left_im = cv2.remap(left_im,
    #             xmap_left,
    #             ymap_left,
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

    # Project points into camera frame
    xyz_homog = cv2.convertPointsToHomogeneous(xyz).squeeze(1).T
    camera_frame_points = (np.linalg.inv(lidar_to_left_cam) @ xyz_homog)[:3].T
    valid_points = camera_frame_points[:,2] > 0
    camera_frame_points = camera_frame_points[valid_points]
    lidar_frame_points = xyz[valid_points]

    # Project into image frame
    H = np.linalg.inv(lidar_to_left_cam)
    image_frame_points = cv2.projectPoints(lidar_frame_points, cv2.Rodrigues(H[:3,:3])[0], H[:3,3], K_left[:3,:3], distortion_left)[0].squeeze(1).astype(int)
    # image_frame_points_homog = K_left[:3,:3] @ camera_frame_points.T
    # image_frame_points = cv2.convertPointsFromHomogeneous(image_frame_points_homog.T).squeeze(1).astype(int)

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