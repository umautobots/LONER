from pathlib import Path
import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bayes_opt import BayesianOptimization
from geometry_msgs.msg import TransformStamped
import tf2_py
import tqdm
import rospy
import ros_numpy
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation
from utils import *


MIN_DISPARITY = 0
NUM_DISPARITIES = 4
BLOCK_SIZE = 4
P1 = 8
P2 = 32
DISP_12_MAX_DIFF = 1
PRE_FILTER_CAP = 31
UNIQUENESS_RATIO = 10
SPECKLE_RANGE = 4
SPECKLE_WINDOW_SIZE = 1000



MAX_RANGE = 250


parser = argparse.ArgumentParser("stereo to rgbd")
parser.add_argument("rosbag_path", type=str)
parser.add_argument("calib_path", type=str)
parser.add_argument("--gui", action="store_true",default=False)
parser.add_argument("--autotune", action="store_true",default=False)

args = parser.parse_args()

bridge = CvBridge()

### Adapted from https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
calibration = FusionPortableCalibration(args.calib_path)

K_left = calibration.left_cam_intrinsic["K"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
rect_left = calibration.left_cam_intrinsic["rectification_matrix"]
size_left = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])

xmap_left, ymap_left = cv2.initUndistortRectifyMap(K_left, distortion_left, rect_left, K_left, size_left, cv2.CV_32FC1)
xmap_inv_left, ymap_inv_left = cv2.initInverseRectificationMap(K_left, distortion_left, rect_left, K_left, size_left, cv2.CV_32FC1)

K_right = calibration.right_cam_intrinsic["K"]
distortion_right = calibration.right_cam_intrinsic["distortion_coeffs"]
rect_right = calibration.right_cam_intrinsic["rectification_matrix"]
size_right = (calibration.right_cam_intrinsic["width"], calibration.right_cam_intrinsic["height"])

xmap_right, ymap_right = cv2.initUndistortRectifyMap(K_right, distortion_right, rect_right, K_right, size_right, cv2.CV_32FC1)
xmap_inv_right, ymap_inv_right = cv2.initInverseRectificationMap(K_right, distortion_right, rect_right, K_right, size_right, cv2.CV_32FC1)

bag = rosbag.Bag(args.rosbag_path)

bag_it = bag.read_messages(topics=["/os_cloud_node/points"])

lidar_timestamps = []
for _,_,ts in bag_it:
    lidar_timestamps.append(ts.to_sec())
lidar_timestamps = np.stack(lidar_timestamps)

bag_it = bag.read_messages(topics=["/stereo/frame_left/image_raw", "/stereo/frame_right/image_raw"])


images = []

prev_ts = -1
while True:
    try:
        topic1, msg1, timestamp1 = next(bag_it)
        _, msg2, timestamp2 = next(bag_it)
    except StopIteration:
        break

    assert (timestamp2 - timestamp1).to_sec() < 0.02, "Timestamps too far apart"

    ts = timestamp1.to_sec()
    
    if ts - prev_ts < 3:
        continue

    prev_ts = ts

    if "left" in topic1:
        left_im = msg1
        right_im = msg2
    else:
        left_im = msg2
        right_im = msg1

    left_im_cv2 = bridge.imgmsg_to_cv2(left_im)
    right_im_cv2 = bridge.imgmsg_to_cv2(right_im)

    images.append((left_im_cv2, right_im_cv2, timestamp1))

kept_lidar_timestamps = []
for _,_,timestamp in images:
    closest_timestamp = np.argmin(np.abs(timestamp.to_sec() - lidar_timestamps))
    kept_lidar_timestamps.append(lidar_timestamps[closest_timestamp])

lidar_scans = []
ts_idx = 0
for _,msg,timestamp in bag.read_messages(topics=["/os_cloud_node/points"]):
    if abs(timestamp.to_sec() - kept_lidar_timestamps[ts_idx]) < 0.005:
        ts_idx += 1
        lidar_scans.append(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg))
    if ts_idx == len(kept_lidar_timestamps):
        break



timestamps = [i[2] for i in images]

lidar_to_cam = calibration.t_lidar_to_left_cam

xyz = lidar_to_cam["xyz"]
xyz = np.array(xyz).reshape(3,1)
rotation = lidar_to_cam["orientation"]
rotation[0], rotation[3] = rotation[3], rotation[0]
rotmat = Rotation.from_quat(rotation).as_matrix()
lidar_to_cam = np.hstack((rotmat, xyz))
lidar_to_cam = np.vstack((lidar_to_cam, [0,0,0,1]))  

# Get ground truth trajectory
rosbag_path = Path(args.rosbag_path)
ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
tf_buffer, _ = build_buffer_from_df(ground_truth_df)
lidar_poses = build_poses_from_buffer(tf_buffer, timestamps)

images = [(images[i][0], images[i][1], lidar_poses[i]) for i in range(len(images)) if lidar_poses[i] is not None]

ground_truth_map_file = "./ground_truth_mesh_coarse.pcd"
ground_truth_map = o3d.io.read_point_cloud(ground_truth_map_file.as_posix())
ground_truth_points = np.asarray(ground_truth_map.points)

depth_images = []
print("Generating Depth Images")

for idx, (left_im, _, lidar_pose) in tqdm.tqdm(enumerate(images[:1])):

    pose = lidar_pose @ lidar_to_cam
    H = np.linalg.inv(pose)

    proj_mat = K_left @ H[:3]

    ground_truth_points_homog = np.hstack((ground_truth_points, np.ones_like(ground_truth_points[:,0]).reshape(-1,1))).T
    all_depths = np.linalg.norm(pose[:3,3] - ground_truth_points, axis=1)

    image_coords = proj_mat @ ground_truth_points_homog

    valid_coords = image_coords[2] > 0
    
    image_coords = image_coords[:,valid_coords]
    image_coords /= image_coords[2]
    projected_points = image_coords[:2].T
    all_depths = all_depths[valid_coords]


    depth_image = np.zeros_like(left_im[...,0])
    
    good_width = np.logical_and(0 <= projected_points[:,0], projected_points[:,0] < depth_image.shape[1])
    good_height = np.logical_and(0 <= projected_points[:,1], projected_points[:,1] < depth_image.shape[0])
    good_items = np.logical_and(good_width, good_height)
    good_items = np.logical_and(good_items, all_depths > 0)

    projected_points = projected_points[good_items].astype(int)
    projected_depths = all_depths[good_items]

    # sort from closest to furthest, so that we can get rid of non-unique points
    sorted_indices = np.argsort(projected_depths)
    projected_depths = projected_depths[sorted_indices]
    projected_points = projected_points[sorted_indices]

    _, unique_indices = np.unique(projected_points[:,:2], axis = 0, return_index=True)
    projected_points = projected_points[unique_indices]
    projected_depths = projected_depths[unique_indices]

    depth_image[projected_points[:,1], projected_points[:,0]] = projected_depths

    depth_images.append(depth_image)

if args.gui:
    def nothing(x):
        pass

    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)

    cv2.createTrackbar('minDisparity','disp',MIN_DISPARITY,25,nothing)
    cv2.createTrackbar('numDisparities','disp',NUM_DISPARITIES,17,nothing)
    cv2.createTrackbar('blockSize','disp',BLOCK_SIZE,50,nothing)
    cv2.createTrackbar('P1','disp',P1,32,nothing)
    cv2.createTrackbar('P2','disp',P2,100,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',DISP_12_MAX_DIFF,150,nothing)
    cv2.createTrackbar('preFilterCap','disp',PRE_FILTER_CAP,62,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',UNIQUENESS_RATIO,25,nothing)
    cv2.createTrackbar('speckleRange','disp',SPECKLE_RANGE,50,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',SPECKLE_WINDOW_SIZE,2500,nothing)
    cv2.createTrackbar('image','disp',0,len(images), nothing)

    # Creating an object of StereoBM algorithm
    stereo: cv2.StereoSGBM = cv2.StereoSGBM_create()

    minDisparity = MIN_DISPARITY
    numDisparities = NUM_DISPARITIES
    blockSize = BLOCK_SIZE
    p1 = P1
    p2 = P2
    disp12MaxDiff = DISP_12_MAX_DIFF
    preFilterCap = PRE_FILTER_CAP
    uniquenessRatio = UNIQUENESS_RATIO
    speckleRange = SPECKLE_RANGE
    speckleWindowSize = SPECKLE_WINDOW_SIZE

    while True:
        imgL, imgR, pose = images[cv2.getTrackbarPos('image','disp')]
        depth_image = depth_images[cv2.getTrackbarPos('image','disp')].copy()

        # Applying stereo image rectification on the left image
        Left_nice= cv2.remap(imgL,
                xmap_left,
                ymap_left,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR,
                xmap_right,
                ymap_right,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)

        # Updating the parameters based on the trackbar positions
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')
        p1 = cv2.getTrackbarPos('P1','disp')*3*blockSize**2
        p2 = cv2.getTrackbarPos('P2','disp')*3*blockSize**2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        
        # Setting the updated parameters before computing disparity map
        stereo.setMinDisparity(minDisparity)
        stereo.setNumDisparities(max(numDisparities, 16))
        stereo.setBlockSize(blockSize)
        stereo.setP1(p1)
        stereo.setP2(p2)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)


        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32 
        disparity = disparity.astype(np.float32) / 16.0

        # Scaling down the disparity values and normalizing them 
        # disparity = (disparity/16.0 - minDisparity)/numDisparities

        ptcloud = cv2.reprojectImageTo3D(disparity, calibration.stereo_disp_to_depth_matrix)
        depthmap = np.linalg.norm(ptcloud, axis=2) * 10
        depthmap = np.nan_to_num(depthmap, posinf=0, neginf=0.)
        depthmap[depthmap > MAX_RANGE] = 0

        # depthmap = cv2.remap(depthmap, xmap_inv_left, ymap_inv_left, cv2.INTER_LANCZOS4,
        #                      cv2.BORDER_CONSTANT,0)

        depthmap_viz = cv2.normalize(depthmap, depthmap, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        depthmap_viz[depthmap_viz == 255] = 0
        depthmap_viz = cv2.applyColorMap(depthmap_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

        disparity_viz = (disparity - minDisparity) / numDisparities * 255
        disparity_viz = cv2.applyColorMap(disparity_viz.astype(np.uint8), cv2.COLORMAP_COOL)

        # depth_image_viz = cv2.remap(depth_image, xmap_left, ymap_left, cv2.INTER_LANCZOS4,
        #         cv2.BORDER_CONSTANT,
        #         0)
        depth_image_viz = cv2.normalize(depth_image, depth_image, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        depth_image_viz = cv2.applyColorMap(depth_image_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

        im = np.vstack((np.hstack((disparity_viz, depthmap_viz)), np.hstack((depth_image_viz, imgL))))
        # Displaying the disparity map
        cv2.imshow("disp",im)
        # cv2.imshow("disp",depthmap_viz)

        # ptcloud = ptcloud.reshape(-1, 3)
        # dists = np.linalg.norm(ptcloud, axis=1)
        # ptcloud = ptcloud[dists < MAX_RANGE]

        # est_pcd = o3d.cuda.pybind.geometry.PointCloud()
        # est_pcd.points = o3d.cuda.pybind.utility.Vector3dVector(ptcloud)
        # est_pcd.transform(pose)
        
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break
        
        # o3d.visualization.draw_geometries([est_pcd])

elif args.autotune:


    MIN_DISPARITY = 3
    NUM_DISPARITIES = 2
    BLOCK_SIZE = 5
    P1 = 8
    P2 = 32
    DISP_12_MAX_DIFF = 0
    PRE_FILTER_CAP = 3
    UNIQUENESS_RATIO = 0
    SPECKLE_RANGE = 2
    SPECKLE_WINDOW_SIZE = 100


    intrinsic_o3d = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"], K_left)
    def evaluate(minDisparity, numDisparities, blockSize, p1, p2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleRange, speckleWindowSize):

        stereo: cv2.StereoSGBM = cv2.StereoSGBM_create()
        stereo.setMinDisparity(minDisparity)
        stereo.setNumDisparities(max(numDisparities, 16))
        stereo.setBlockSize(blockSize)
        stereo.setP1(p1)
        stereo.setP2(p2)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)

        for imgL, imgR, pose in images:
            # Applying stereo image rectification on the left image
            Left_nice= cv2.remap(imgL,
                    xmap_left,
                    ymap_left,
                    cv2.INTER_LANCZOS4,
                    cv2.BORDER_CONSTANT,
                    0)
            
            # Applying stereo image rectification on the right image
            Right_nice = cv2.remap(imgR,
                    xmap_right,
                    ymap_right,
                    cv2.INTER_LANCZOS4,
                    cv2.BORDER_CONSTANT,
                    0)


            disparity = stereo.compute(Left_nice,Right_nice)
            disparity = disparity.astype(np.float32)

            # Scaling down the disparity values and normalizing them 
            disparity = (disparity/16.0 - minDisparity)/numDisparities

            ptcloud = cv2.reprojectImageTo3D(disparity, calibration.stereo_disp_to_depth_matrix, handleMissingValues=True)
            
            ptcloud = ptcloud.reshape(-1, 3)
            dists = np.linalg.norm(ptcloud, axis=1)
            ptcloud = ptcloud[dists < MAX_RANGE]

            est_pcd = o3d.cuda.pybind.geometry.PointCloud()
            est_pcd.points = o3d.cuda.pybind.utility.Vector3dVector(ptcloud)
            est_pcd.transform(pose)
            

            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            for geometry in [est_pcd]:
                viewer.add_geometry(geometry)
            opt = viewer.get_render_option()
            opt.show_coordinate_frame = True
            # opt.background_color = np.asarray([0.5, 0.5, 0.5])
            viewer.run()
            viewer.destroy_window()

    evaluate(MIN_DISPARITY, NUM_DISPARITIES, BLOCK_SIZE, P1, P2, DISP_12_MAX_DIFF, PRE_FILTER_CAP, UNIQUENESS_RATIO, SPECKLE_RANGE, SPECKLE_WINDOW_SIZE)