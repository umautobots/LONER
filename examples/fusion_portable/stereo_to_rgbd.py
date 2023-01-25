from pathlib import Path
import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
import rosbag
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from bayes_opt import BayesianOptimization
from geometry_msgs.msg import TransformStamped, PoseStamped
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import tf2_msgs.msg
import tqdm
import rospy
import ros_numpy
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from utils import *
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
import torch
from torchvision.utils import flow_to_image
from more_itertools import peekable
import numpy.lib.recfunctions as rf 

MIN_DISPARITY = 0
NUM_DISPARITIES = 3
BLOCK_SIZE = 5
P1 = 8
P2 = 32
DISP_12_MAX_DIFF = 1
PRE_FILTER_CAP = 0
UNIQUENESS_RATIO = 10
SPECKLE_RANGE = 2
SPECKLE_WINDOW_SIZE = 100
MODE_HH = True

MAX_RANGE = 150


parser = argparse.ArgumentParser("stereo to rgbd")
parser.add_argument("rosbag_path", type=str)
parser.add_argument("calib_path", type=str)
parser.add_argument("--gui", action="store_true",default=False)
parser.add_argument("--autotune", action="store_true",default=False)
parser.add_argument("--raft", action="store_true",default=False)
parser.add_argument("--build_rosbag", action="store_true", default=False)
parser.add_argument("--log_lidar", action="store_true", default=False)
parser.add_argument("--log_tf2", action="store_true", default=False)

args = parser.parse_args()


bridge = CvBridge()

### Load in the calibration
calibration = FusionPortableCalibration(args.calib_path)
K_left = calibration.left_cam_intrinsic["K"]
proj_left = calibration.left_cam_intrinsic["projection_matrix"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
rect_left = calibration.left_cam_intrinsic["rectification_matrix"]
size_left = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])

xmap_left, ymap_left = cv2.initUndistortRectifyMap(K_left, distortion_left, rect_left, proj_left, size_left, cv2.CV_32FC1)
xmap_inv_left, ymap_inv_left = cv2.initInverseRectificationMap(K_left, distortion_left, rect_left, proj_left, size_left, cv2.CV_32FC1)

K_right = calibration.right_cam_intrinsic["K"]
proj_right = calibration.right_cam_intrinsic["projection_matrix"]
distortion_right = calibration.right_cam_intrinsic["distortion_coeffs"]
rect_right = calibration.right_cam_intrinsic["rectification_matrix"]
size_right = (calibration.right_cam_intrinsic["width"], calibration.right_cam_intrinsic["height"])

xmap_right, ymap_right = cv2.initUndistortRectifyMap(K_right, distortion_right, rect_right, proj_right, size_right, cv2.CV_32FC1)
xmap_inv_right, ymap_inv_right = cv2.initInverseRectificationMap(K_right, distortion_right, rect_right, proj_right, size_right, cv2.CV_32FC1)


lidar_to_cam = calibration.t_lidar_to_left_cam

xyz = lidar_to_cam["xyz"]
xyz = np.array(xyz).reshape(3,1)
rotation = lidar_to_cam["orientation"]
rotation[0], rotation[3] = rotation[3], rotation[0]
rotmat = Rotation.from_quat(rotation).as_matrix()
lidar_to_cam = np.hstack((rotmat, xyz))
lidar_to_cam = np.vstack((lidar_to_cam, [0,0,0,1]))  


weights = Raft_Large_Weights.C_T_V2
raft_model = raft_large(weights=weights, progress=False).cuda()
raft_model = raft_model.eval()
raft_transforms = weights.transforms()

# Get ground truth trajectory
rosbag_path = Path(args.rosbag_path)
ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
tf_buffer, _ = build_buffer_from_df(ground_truth_df)

if args.raft:
    calibration.stereo_disp_to_depth_matrix[3,2] *= -1


if args.gui:
    ### Extract data from rosbag
    bag = rosbag.Bag(args.rosbag_path)

    bag_it = bag.read_messages(topics=["/os_cloud_node/points"])

    ## Pass 1: Get all the lidar timestamps in the bag
    lidar_timestamps = []
    for _,_,ts in bag_it:
        lidar_timestamps.append(ts.to_sec())
    lidar_timestamps = np.stack(lidar_timestamps)

    bag_it = bag.read_messages(topics=["/stereo/frame_left/image_raw", "/stereo/frame_right/image_raw"])

    ## Pass 2: Get left/right image pairs, once every 3 seconds
    images = []
    prev_ts = -1
    kept_lidar_timestamps = []
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
            left_ts = timestamp1
        else:
            left_im = msg2
            right_im = msg1
            left_ts = timestamp2

        left_im_cv2 = bridge.imgmsg_to_cv2(left_im)
        right_im_cv2 = bridge.imgmsg_to_cv2(right_im)


        closest_lidar_timestamp_idx = np.argmin(np.abs(left_ts.to_sec() - lidar_timestamps))
        kept_lidar_timestamps.append(lidar_timestamps[closest_lidar_timestamp_idx])

        images.append((left_im_cv2, right_im_cv2, left_ts))

    # Pass 3: Go back and associate a lidar scan to the left/right image pairs by timestamp
    lidar_scans = []
    actual_lidar_timestamps = []
    ts_idx = 0
    for _,msg,timestamp in bag.read_messages(topics=["/os_cloud_node/points"]):
        if abs(timestamp.to_sec() - kept_lidar_timestamps[ts_idx]) < 0.05:
            try:
                lidar_pose_msg = tf_buffer.lookup_transform_core("map", "lidar", timestamp)
            except:
                lidar_scans.append(None)
                actual_lidar_timestamps.append(None)
                ts_idx += 1
                continue
                
            actual_lidar_timestamps.append(timestamp)
            lidar_pose = msg_to_transformation_mat(lidar_pose_msg)
            # lidar_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)

            # print(timestamp.to_sec())
            # cam_pose = lidar_pose @ lidar_to_cam

            # lidar_points_homog = np.hstack((lidar_points, np.ones_like(lidar_points[:,0:1]))).T
            # lidar_points_global = lidar_pose @ lidar_points_homog
            # lidar_points_global /= lidar_points_global[3]
            # lidar_points_global = lidar_points_global[:3].T

            lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            lidar_data = pd.DataFrame(lidar_data).to_numpy()
            xyz = lidar_data[:, :3]
            
            timestamps = lidar_data[:,-1:].astype(np.float64)

            data = np.hstack((xyz, timestamps))

            lidar_scans.append(data)

            # cam_pts = np.linalg.inv(lidar_to_cam)[:3] @ lidar_points_homog
            # im_points = K_left @ cam_pts
            # depths = np.linalg.norm(cam_pts, axis=0)
            # print(depths.shape, cam_pts.shape)
            # valid_points = im_points[2]>0
            # valid_cam_pts = lidar_points[valid_points]
            # depths = depths[valid_points]
            # cam_to_lidar = np.linalg.inv(lidar_to_cam)
            # im_points_distorted, _ = cv2.projectPoints(valid_cam_pts, cv2.Rodrigues(cam_to_lidar[:3,:3])[0], cam_to_lidar[:3,3],
            #                                         K_left, distortion_left)
            # im_points_distorted = im_points_distorted.squeeze(1)
            # # Check in frame
            # good_points = im_points_distorted[:,0] >= 0
            # good_points = np.logical_and(good_points, im_points_distorted[:,0] < size_left[0])
            # good_points = np.logical_and(good_points, im_points_distorted[:,1] >= 0)
            # good_points = np.logical_and(good_points, im_points_distorted[:,1] < size_left[1])


            # im_points_distorted = im_points_distorted[good_points].astype(int)
            # depths = depths[good_points]

            # depths = cv2.normalize(depths, depths, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
            # depths = cv2.applyColorMap(depths.astype(np.uint8), cv2.COLORMAP_TURBO).squeeze(1)

            # disp_im = images[ts_idx][0].copy()
            # disp_im[im_points_distorted[:,1],im_points_distorted[:,0]] = depths

            # cv2.imshow("im", disp_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            ts_idx += 1

        if ts_idx == len(kept_lidar_timestamps):
            break

    image_timestamps = [i[2] for i in images]

    lidar_poses = build_poses_from_buffer(tf_buffer, image_timestamps)
    images = [(images[i][0], images[i][1], lidar_poses[i], lidar_scans[i], image_timestamps[i]) \
        for i in range(len(images)) if lidar_poses[i] is not None and lidar_scans[i] is not None]

    # ground_truth_map_file = rosbag_path.parent / "ground_truth_map.pcd"
    # ground_truth_map = o3d.io.read_point_cloud(ground_truth_map_file.as_posix())
    # ground_truth_map = o3d.io.read_point_cloud("./ground_truth_mesh_coarse.pcd")
    # ground_truth_points = np.asarray(ground_truth_map.points)

    # depth_images = []
    compensated_scans = []
    print("Generating Depth Images")

    for idx, (left_im, _, lidar_pose, lidar_scan, ts) in tqdm.tqdm(enumerate(images)):

        start_ts = ts.to_sec()
        end_ts = ts  + rospy.Duration.from_sec(np.max(lidar_scan[:,-1]).astype(np.float128))
        

        try:
            start_pose = tf_buffer.lookup_transform_core("map","lidar",ts)
            end_pose = tf_buffer.lookup_transform_core("map","lidar",end_ts)

            start_pose = msg_to_transformation_mat(start_pose)
            end_pose = msg_to_transformation_mat(end_pose)
        except:
            print("bad depth image poses")
            # depth_images.append(None)
            continue
        
        scan_duration = np.max(lidar_scan[:,-1])
        rotmats = np.concatenate([np.expand_dims(start_pose[:3,:3],axis=0), np.expand_dims(end_pose[:3,:3], axis=0)], axis=0)
        rots = Rotation.from_matrix(rotmats)
        xyzs = [start_pose[:3,3], end_pose[:3,3]]
        xyzs = np.stack(xyzs)
        slerp = Slerp([0, scan_duration], rots)
        xyz_interp = interp1d([0, scan_duration], xyzs, axis=0)

        timestamps = lidar_scan[:,-1]
        rotations = slerp(timestamps).as_matrix()
        translations = xyz_interp(timestamps).reshape(-1, 3, 1)
        T = np.concatenate((rotations, translations), axis=2)
        T = np.hstack((T, np.tile([0,0,0,1], (translations.shape[0],1,1))))
        xyz = lidar_scan[:,:3]
        xyz_homog = np.hstack((xyz, np.ones_like(xyz[:,0:1]))).reshape(-1, 4, 1)
        ground_truth_points = T @ xyz_homog
        ground_truth_points = ground_truth_points.squeeze(2)[:,:3]

        compensated_scans.append(ground_truth_points)

        # # Project into the rectified image frame
        # proj_mat = proj_left[:3,:3] @ H[:3]

        # # First do the projection manually just to figure out what's in front of the camera
        # ground_truth_points_homog = np.hstack((ground_truth_points, np.ones_like(ground_truth_points[:,0:1]))).T
        # all_depths = np.linalg.norm(pose[:3,3] - ground_truth_points, axis=1)
        # image_coords = proj_mat @ ground_truth_points_homog
        # valid_coords = image_coords[2] > 0
        # image_coords = image_coords[:,valid_coords]
        # image_coords /= image_coords[2]
        # projected_points = image_coords[:2].T
        # all_depths = all_depths[valid_coords]


        # depth_image = np.zeros_like(left_im[...,0])
        
        # good_width = np.logical_and(0 <= projected_points[:,0], projected_points[:,0] < depth_image.shape[1])
        # good_height = np.logical_and(0 <= projected_points[:,1], projected_points[:,1] < depth_image.shape[0])
        # good_items = np.logical_and(good_width, good_height)
        # good_items = np.logical_and(good_items, all_depths > 0)
        # good_items = np.logical_and(good_items, all_depths < 25)

        # projected_points = projected_points[good_items].astype(int)
        # projected_depths = all_depths[good_items]

        # # sort from closest to furthest, so that we can get rid of non-unique points
        # sorted_indices = np.argsort(projected_depths)
        # projected_depths = projected_depths[sorted_indices]
        # projected_points = projected_points[sorted_indices]

        # _, unique_indices = np.unique(projected_points[:,:2], axis = 0, return_index=True)
        # projected_points = projected_points[unique_indices]
        # projected_depths = projected_depths[unique_indices]



        # projected_depths_viz = cv2.normalize(projected_depths, projected_depths.copy(), alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        # projected_depths_viz = cv2.applyColorMap(projected_depths_viz.astype(np.uint8), cv2.COLORMAP_TURBO).squeeze(1)
        # projected_depths_viz = np.hstack((projected_depths_viz, 0.5 * np.ones_like(projected_depths_viz[:,0:1])))

        # disp_im = cv2.remap(left_im,xmap_left,ymap_left,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)
        # disp_im = cv2.cvtColor(disp_im, cv2.COLOR_BGR2BGRA)
        # disp_im[:,:,3] = disp_im[:,:,3] //2
        
        # disp_im[projected_points[:,1],projected_points[:,0]] = projected_depths_viz

        # cv2.imshow("im", disp_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # depth_image[projected_points[:,1], projected_points[:,0]] = projected_depths

        # depth_images.append(depth_image)
    ### Adapted from https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

    def nothing(x):
        pass

    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)

    if not args.raft:
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
        imgL, imgR, pose, _, _= images[cv2.getTrackbarPos('image','disp')]
        idx = cv2.getTrackbarPos('image','disp')
        # depth_image = depth_images[cv2.getTrackbarPos('image','disp')].copy()
        # min_depth = np.min(depth_image)
        # max_depth = np.max(depth_image)
        # gt_depth_image_viz = cv2.normalize(depth_image, depth_image, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        # gt_depth_image_viz = cv2.applyColorMap(gt_depth_image_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

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

        if args.raft:
            left_im = torch.from_numpy(Left_nice).permute(2,0,1).unsqueeze(0)
            right_im = torch.from_numpy(Right_nice).permute(2,0,1).unsqueeze(0)
            left_batch, right_batch = raft_transforms(left_im, right_im)
            all_flows = raft_model(left_batch.cuda(), right_batch.cuda())
            predicted_flows = all_flows[-1]

            left_im = left_im.squeeze(0).permute(1,2,0).cpu().numpy()
            right_im = right_im.squeeze(0).permute(1,2,0).cpu().numpy()
            disparity = predicted_flows[0,0,:].detach().cpu().numpy()
            disp_copy = disparity.copy()
            yflows = predicted_flows[0,1,:].detach().cpu().numpy()

            disparity_viz = cv2.normalize(disparity, disp_copy, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
            disparity_viz = cv2.applyColorMap(disparity_viz.astype(np.uint8), cv2.COLORMAP_TURBO)
            yflows_viz = cv2.normalize(yflows, yflows, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
            yflows_viz = cv2.applyColorMap(yflows_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

            ptcloud = cv2.reprojectImageTo3D(disparity, calibration.stereo_disp_to_depth_matrix)
            depth_image = np.linalg.norm(ptcloud, axis=2)
            depth_image = np.nan_to_num(depth_image, posinf=0, neginf=0.)
            depth_image_viz = cv2.normalize(depth_image, depth_image, alpha=255,beta=0,norm_type=cv2.NORM_MINMAX)
            depth_image_viz = cv2.applyColorMap(depth_image_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

            disparity_unrectified = cv2.remap(disparity_viz,
                xmap_inv_left, ymap_inv_left, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            depth_unrectified = cv2.remap(depth_image_viz,
                xmap_inv_left, ymap_inv_left, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

            im = np.vstack(( np.hstack((imgL, imgR)),np.hstack((disparity_unrectified, depth_unrectified))))#gt_depth_image_viz))))
        else:
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
            if MODE_HH:
                stereo.setMode(cv2.StereoSGBM_MODE_HH)


            # Calculating disparity using the StereoBM algorithm
            disparity = stereo.compute(Left_nice,Right_nice)
            # NOTE: Code returns a 16bit signed single channel image,
            # CV_16S containing a disparity map scaled by 16. Hence it 
            # is essential to convert it to CV_32F and scale it down 16 times.

            # Converting to float32 
            disparity = disparity.astype(np.float32) / 16.0

            ptcloud = cv2.reprojectImageTo3D(disparity, calibration.stereo_disp_to_depth_matrix)
            depthmap = np.linalg.norm(ptcloud, axis=2)
            depthmap = np.nan_to_num(depthmap, posinf=0, neginf=0.)
            depthmap[depthmap > MAX_RANGE] = 0

            # depthmap = cv2.remap(depthmap, xmap_inv_left, ymap_inv_left, cv2.INTER_LANCZOS4,
            #                      cv2.BORDER_CONSTANT,0)


            disparity_viz = (disparity - minDisparity) / numDisparities * 255
            disparity_viz = cv2.applyColorMap(disparity_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

            # depthmap_viz = 255 * (depthmap - min_depth)/(max_depth-min_depth)
            # depthmap_viz = np.clip(depthmap_viz, 0, 255)
            
            depthmap_viz = cv2.normalize(depthmap, depthmap, alpha=255,beta=0,norm_type=cv2.NORM_MINMAX)
            depthmap_viz = cv2.applyColorMap(depthmap_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

            left_viz = Left_nice.copy()
            scan = compensated_scans[cv2.getTrackbarPos('image','disp')]
            scan_homog = np.hstack((scan, np.ones_like(scan[:,0:1]))).T
            H = np.linalg.inv(pose @ lidar_to_cam)
            im_points = proj_left[:3,:3] @ H[:3] @ scan_homog
            valid_points = im_points[2]>0

            im_points = im_points[:, valid_points].T
            im_points = im_points / im_points[:,2:]
            depths = np.linalg.norm(scan[valid_points], axis=1)

            # Check in frame
            good_points = im_points[:,0] >= 0
            good_points = np.logical_and(good_points, im_points[:,0] < size_left[0])
            good_points = np.logical_and(good_points, im_points[:,1] >= 0)
            good_points = np.logical_and(good_points, im_points[:,1] < size_left[1])
            
            im_points = im_points[good_points].astype(int)
            depths = depths[good_points]
            
            depths = cv2.normalize(depths, depths, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
            depths = cv2.applyColorMap(depths.astype(np.uint8), cv2.COLORMAP_TURBO).squeeze(1)

            left_viz[im_points[:,1],im_points[:,0]] = depths

            im = np.vstack((np.hstack((disparity_viz, depthmap_viz)), 
                            # np.hstack((gt_depth_image_viz, gt_depth_image_viz)),
                            np.hstack((left_viz, Right_nice))))
        # Displaying the disparity map
        cv2.imshow("disp",im)
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break
        
        # o3d.visualization.draw_geometries([est_pcd])

elif args.build_rosbag:
    rosbag_path = Path(args.rosbag_path)
    bag = rosbag.Bag(rosbag_path.as_posix())

    if not args.raft:
        stereo: cv2.StereoSGBM = cv2.StereoSGBM_create()

        stereo.setMinDisparity(MIN_DISPARITY)
        stereo.setNumDisparities(max(NUM_DISPARITIES, 16))
        stereo.setBlockSize(BLOCK_SIZE)
        stereo.setP1(P1)
        stereo.setP2(P2)
        stereo.setDisp12MaxDiff(DISP_12_MAX_DIFF)
        stereo.setPreFilterCap(PRE_FILTER_CAP)
        stereo.setUniquenessRatio(UNIQUENESS_RATIO)
        stereo.setSpeckleRange(SPECKLE_RANGE)
        stereo.setSpeckleWindowSize(SPECKLE_WINDOW_SIZE)
        if MODE_HH:
            stereo.setMode(cv2.StereoSGBM_MODE_HH)

    if args.log_tf2:
        t_lc = lidar_to_cam[:3,3]
        r_lc = lidar_to_cam[:3,:3]

        quat_lc = Rotation.from_matrix(r_lc).as_quat()

        lidar_to_cam_msg = TransformStamped()
        lidar_to_cam_msg.header.frame_id = "lidar"
        lidar_to_cam_msg.child_frame_id = "camera/depth"
        
        lidar_to_cam_msg.transform.translation.x = t_lc[0]
        lidar_to_cam_msg.transform.translation.y = t_lc[1]
        lidar_to_cam_msg.transform.translation.z = t_lc[2]
        
        lidar_to_cam_msg.transform.rotation.x = quat_lc[0]
        lidar_to_cam_msg.transform.rotation.y = quat_lc[1]
        lidar_to_cam_msg.transform.rotation.z = quat_lc[2]
        lidar_to_cam_msg.transform.rotation.w = quat_lc[3]

    cam_info_msg = CameraInfo()
    cam_info_msg.header.frame_id = "/camera/depth"
    cam_info_msg.width = size_left[0]
    cam_info_msg.height = size_left[1]
    cam_info_msg.distortion_model = "plumb_bob"
    cam_info_msg.D = distortion_left.tolist()
    cam_info_msg.K = K_left.flatten().tolist()
    cam_info_msg.R = rect_left.flatten().tolist()
    cam_info_msg.P = proj_left.flatten().tolist()
    

    bag_it = peekable(bag.read_messages(topics=["/stereo/frame_left/image_raw", "/stereo/frame_right/image_raw", "/os_cloud_node/points"]))
    
    output_bag = rosbag.Bag("canteen_rgbd.bag", 'w')
    
    for idx in tqdm.trange(bag.get_message_count(["/stereo/frame_left/image_raw", "/stereo/frame_right/image_raw"])//2):
        # if idx > 100:
        #     break
        try:
            # This relies on the fact the the bags are very well-ordered: 1 lidar, then 4 cameras, forever.
            
            while bag_it.peek()[0] == "/os_cloud_node/points":
                lidar_topic, lidar_msg, lidar_ts = next(bag_it)
                lidar_msg.header.frame_id = "lidar"
            
            topic1, msg1, timestamp1 = next(bag_it)

            while bag_it.peek()[0] == "/os_cloud_node/points":
                lidar_topic, lidar_msg, lidar_ts = next(bag_it)
                lidar_msg.header.frame_id = "lidar"
            
            _, msg2, timestamp2 = next(bag_it)

            if bag_it.peek()[0] == "/os_cloud_node/points":
                lidar_topic, lidar_msg, lidar_ts = next(bag_it)
                lidar_msg.header.frame_id = "lidar"
        except StopIteration:
            break

        assert (timestamp2 - timestamp1).to_sec() < 0.02, "Timestamps too far apart"

        if "left" in topic1:
            left_im = bridge.imgmsg_to_cv2(msg1)
            right_im = bridge.imgmsg_to_cv2(msg2)
            left_msg = msg1
            left_ts = timestamp1
        else:
            left_im = bridge.imgmsg_to_cv2(msg2)
            right_im = bridge.imgmsg_to_cv2(msg1)
            left_msg = msg2
            left_ts = timestamp2

        # Applying stereo image rectification on the left image
        left_rect = cv2.remap(left_im,
                xmap_left,
                ymap_left,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        right_rect = cv2.remap(right_im,
                xmap_right,
                ymap_right,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)

        # combined_image = np.hstack((left_rect, right_rect))
        # for line_level in np.arange(50, combined_image.shape[0], 50):
        #     combined_image = cv2.line(combined_image, (0, line_level), (combined_image.shape[1], line_level), (255, 0, 0), 3)
        # cv2.imshow("rectification", combined_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if args.raft:
            left_im = torch.from_numpy(left_rect).permute(2,0,1).unsqueeze(0)
            right_im = torch.from_numpy(right_rect).permute(2,0,1).unsqueeze(0)
            left_batch, right_batch = raft_transforms(left_im, right_im)
            all_flows = raft_model(left_batch.cuda(), right_batch.cuda())
            predicted_flows = all_flows[-1]

            left_im = left_im.squeeze(0).permute(1,2,0).cpu().numpy()
            right_im = right_im.squeeze(0).permute(1,2,0).cpu().numpy()
            disparity = predicted_flows[0,0,:].detach().cpu().numpy()

        else:

            disparity = stereo.compute(left_rect,right_rect)
            # NOTE: Code returns a 16bit signed single channel image,
            # CV_16S containing a disparity map scaled by 16. Hence it 
            # is essential to convert it to CV_32F and scale it down 16 times.

            # Converting to float32 
            disparity = disparity.astype(np.float32) / 16.0

        if args.log_lidar:
            lidar_bag_ts = lidar_ts
        else:
            lidar_bag_ts = left_ts
        
        try:
            world_to_lidar_msg = tf_buffer.lookup_transform_core("map","lidar",lidar_bag_ts)

            if args.log_tf2:
                lidar_to_cam_msg.header.stamp = left_ts
                lidar_to_cam_msg.header.seq = idx
                tf_message = tf2_msgs.msg.TFMessage()

                tf_message.transforms.append(lidar_to_cam_msg)
                world_to_lidar_msg.header.stamp = lidar_bag_ts
                world_to_lidar_msg.header.seq = idx
                tf_message.transforms.append(world_to_lidar_msg)
                output_bag.write("/tf", tf_message, t=lidar_bag_ts)
            if args.log_lidar:
                output_bag.write(lidar_topic, lidar_msg, t=lidar_bag_ts)

            # Log the poses
            world_to_lidar_msg_cam_time = tf_buffer.lookup_transform_core("map","lidar",left_ts)
            world_to_lidar = msg_to_transformation_mat(world_to_lidar_msg_cam_time)
            world_to_cam = world_to_lidar @ lidar_to_cam
            pose_msg_cam = transformation_mat_to_pose_msg(world_to_cam)
            pose_msg_lidar = transformation_mat_to_pose_msg(world_to_lidar)
            pose_msg_lidar_stamped = PoseStamped()
            pose_msg_lidar_stamped.pose = pose_msg_lidar
            pose_msg_lidar_stamped.header.frame_id = "map"
            pose_msg_lidar_stamped.header.stamp = left_ts

            pose_msg_camera_stamped = PoseStamped()
            pose_msg_camera_stamped.pose = pose_msg_cam
            pose_msg_camera_stamped.header.frame_id = "map"
            pose_msg_camera_stamped.header.stamp = left_ts
            output_bag.write("/pose_lidar", pose_msg_lidar_stamped, t=left_ts)
            output_bag.write("/pose_camera", pose_msg_camera_stamped, t=left_ts)
        except Exception as e:
            print("Ignoring: ", e)
        # disp_copy = disparity.copy()
        # yflows = predicted_flows[0,1,:].detach().cpu().numpy()

        # disparity_viz = cv2.normalize(disparity, disp_copy, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        # disparity_viz = cv2.applyColorMap(disparity_viz.astype(np.uint8), cv2.COLORMAP_TURBO)
        # yflows_viz = cv2.normalize(yflows, yflows, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        # yflows_viz = cv2.applyColorMap(yflows_viz.astype(np.uint8), cv2.COLORMAP_TURBO)

        ptcloud = cv2.reprojectImageTo3D(disparity, calibration.stereo_disp_to_depth_matrix)
        ptcloud_data = rf.unstructured_to_structured(ptcloud.reshape(-1, 3), \
            dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)]))

        ptcloud2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(ptcloud_data)
        ptcloud2_msg.header.stamp = left_ts
        ptcloud2_msg.header.frame_id = "camera/depth"

        depth_image = ptcloud[...,2] # np.linalg.norm(ptcloud, axis=2)

        depth_image_unrectified = cv2.remap(depth_image,
            xmap_inv_left, ymap_inv_left, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        depth_image[depth_image > MAX_RANGE] = MAX_RANGE
        depthmsg = bridge.cv2_to_imgmsg(depth_image)
        depthmsg.header.stamp = timestamp1
        depthmsg.header.frame_id = "camera/depth"

        cam_info_msg.header.stamp = left_ts
        output_bag.write("/camera/rgb/camera_info", cam_info_msg, t=left_ts)
        output_bag.write("/camera/depth/camera_info", cam_info_msg, t=left_ts)
        left_msg.header.frame_id = "camera/depth"
        output_bag.write("/camera/rgb/image_raw", left_msg, t=left_ts)
        output_bag.write("/camera/depth", depthmsg, t=left_ts)
        output_bag.write("/depth_cloud", ptcloud2_msg, t=left_ts)
    output_bag.close()