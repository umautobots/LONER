import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
from cv_bridge import CvBridge
import pytorch3d.transforms
import open3d as o3d
from scipy.spatial.transform import Rotation
from utils import *

LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf

parser = argparse.ArgumentParser("Verify Calibration")
parser.add_argument("pcd_path", type=str)
parser.add_argument("im_path", type=str)
parser.add_argument("calib_path", type=str)
parser.add_argument("--im_compressed", action="store_true", default=False, help="Set if using a rosbag with compressed images")

args = parser.parse_args()

bridge = CvBridge()

### Load in the calibration
calibration = FusionPortableCalibration(args.calib_path)
K_left = calibration.left_cam_intrinsic["K"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
im_width, im_height = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])

lidar_to_left_cam = calibration.t_lidar_to_left_cam
xyz = lidar_to_left_cam["xyz"]
xyz = np.array(xyz).reshape(3,1)
rotation = lidar_to_left_cam["orientation"]
rotation = [rotation[i] for i in [1,2,3,0]]
rotmat = Rotation.from_quat(rotation).as_matrix()
lidar_to_left_cam = np.hstack((rotmat, xyz))
lidar_to_left_cam = np.vstack((lidar_to_left_cam, [0,0,0,1]))  

### Load GT poses

compressed_str = "/compressed" if args.im_compressed else ""
cam_topics = [f"/stereo/frame_left/image_raw{compressed_str}", f"/stereo/frame_right/image_raw{compressed_str}"]
all_topics = cam_topics + ["/os_cloud_node/points"]

### Go through bag and get images and corresponding nearby lidar scans

left_im = cv2.imread(args.im_path)

lidar_points = np.asarray(o3d.io.read_point_cloud(args.pcd_path).points)
dists = np.linalg.norm(lidar_points, axis=1)
valid_ranges = dists > LIDAR_MIN_RANGE
lidar_points = lidar_points[valid_ranges]

# Project points into camera frame
lidar_points_homog = cv2.convertPointsToHomogeneous(lidar_points).squeeze(1).T

H = np.linalg.inv(lidar_to_left_cam)

camera_frame_points = (H @ lidar_points_homog)[:3].T
valid_points = camera_frame_points[:,2] > 0
camera_frame_points = camera_frame_points[valid_points]

image_frame_points = cv2.projectPoints(lidar_points[valid_points].T, cv2.Rodrigues(H[:3,:3])[0], H[:3,3], K_left, distortion_left)[0].squeeze(1).astype(int)

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
depth_im = cv2.dilate(depth_im, np.ones((3,3)))
color_coords = depth_im[:,:] != np.zeros(3)
disp_im[color_coords] = depth_im[color_coords]


cv2.imshow("Projected Points", disp_im)
cv2.waitKey(0)
cv2.destroyAllWindows()