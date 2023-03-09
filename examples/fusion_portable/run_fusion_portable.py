#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import shutil
import sys
import re
import pathlib
import time

import cv2
import pandas as pd
import ros_numpy
import rosbag
import rospy
import tf2_py
import torch
from attrdict import AttrDict
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from pathlib import Path
from sensor_msgs.msg import Image, PointCloud2
import pytorch3d.transforms
import torch.multiprocessing as mp

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

IM_SCALE_FACTOR = 0.5

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from fusion_portable_calibration import FusionPortableCalibration

from src.cloner_slam import ClonerSLAM
from src.common.pose import Pose
from src.common.sensors import Image, LidarScan
from src.common.settings import Settings

# autopep8: on

LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf


bridge = CvBridge()

def build_scan_from_msg(lidar_msg: PointCloud2, timestamp: rospy.Time) -> LidarScan:
    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(
        lidar_msg)

    lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy())
    xyz = lidar_data[:, :3]
    
    dists = torch.linalg.norm(xyz, dim=1)
    valid_ranges = dists > LIDAR_MIN_RANGE


    xyz = xyz[valid_ranges].T
    timestamps = (lidar_data[valid_ranges, -1] + timestamp.to_sec()).float()

    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:, indices]

    return LidarScan(directions.float(), dists.float(), timestamps.float())


def tf_to_settings(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    xyz = [trans.x, trans.y, trans.z]
    quat = [rot.w, rot.x, rot.y, rot.z]

    return AttrDict({"xyz": xyz, "orientation": quat})


def build_image_from_msg(image_msg, timestamp) -> Image:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_img = cv2.resize(cv_img, (0,0), fx=IM_SCALE_FACTOR, fy=IM_SCALE_FACTOR)
    pytorch_img = torch.from_numpy(cv_img / 255)
    return Image(pytorch_img, timestamp.to_sec())


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3, 1)
    quat = torch.Tensor([rot.w, rot.x, rot.y, rot.z])
    rotmat = pytorch3d.transforms.quaternion_to_matrix(quat)
    
    T = torch.hstack((rotmat, xyz))
    T = torch.vstack((T, torch.Tensor([0, 0, 0, 1])))
    return T.float()

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

def build_poses_from_buffer(tf_buffer, timestamps, cam_to_lidar):
    lidar_poses = []
    for t in timestamps:
        try:
            lidar_tf = tf_buffer.lookup_transform_core('map', "lidar", t)
            lidar_poses.append(msg_to_transformation_mat(lidar_tf))# @ cam_to_lidar)
        except Exception as e:
            print("Skipping invalid tf")

    return torch.stack(lidar_poses).float()



def run_trial(args, settings, settings_description = None, idx = None):
    init = False
    prev_time = None

    init_clock = time.time()

    calibration = FusionPortableCalibration(args.calibration_path)

    tf_buffer = tf2_py.BufferCore(rospy.Duration(10000))

    ego_pose_timestamps = []

    camera = settings.ros_names.camera
    lidar = settings.ros_names.lidar
    camera_suffix = settings.ros_names.camera_suffix
    topic_prefix = settings.ros_names.topic_prefix

    camera_info_topic = f"{topic_prefix}/{camera}/camera_info" 
    image_topic = f"{topic_prefix}/{camera}/{camera_suffix}"
    lidar_topic = f"{topic_prefix}/{lidar}"

    K = torch.from_numpy(calibration.left_cam_intrinsic["K"]).float()
    K[:2, :] *= IM_SCALE_FACTOR

    settings["calibration"]["camera_intrinsic"]["k"] = K

    settings["calibration"]["camera_intrinsic"]["distortion"] = \
        torch.from_numpy(calibration.left_cam_intrinsic["distortion_coeffs"]).float()

    settings["calibration"]["camera_intrinsic"]["width"] = int(calibration.left_cam_intrinsic["width"] // (1/IM_SCALE_FACTOR))
    
    settings["calibration"]["camera_intrinsic"]["height"] = int(calibration.left_cam_intrinsic["height"] // (1/IM_SCALE_FACTOR))

    ray_range = settings.mapper.optimizer.model_config.data.ray_range
    image_size = (settings.calibration.camera_intrinsic.height,
                settings.calibration.camera_intrinsic.width)

    lidar_to_camera = Pose.from_settings(calibration.t_lidar_to_left_cam).get_transformation_matrix()
    camera_to_lidar = lidar_to_camera.inverse()

    settings["calibration"]["lidar_to_camera"] = calibration.t_lidar_to_left_cam
    settings["experiment_name"] = args.experiment_name

    cloner_slam = ClonerSLAM(settings)

    # Get ground truth trajectory. This is only used to construct the world cube.
    rosbag_path = Path(args.rosbag_path)
    ground_truth_file = rosbag_path.parent / "ground_truth_traj.txt"
    ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    tf_buffer, timestamps = build_buffer_from_df(ground_truth_df)
    lidar_poses = build_poses_from_buffer(tf_buffer, timestamps, camera_to_lidar)
    
    if idx is None:
        ablation_name = None
    else:
        ablation_name = args.experiment_name

    cloner_slam.initialize(camera_to_lidar, lidar_poses, settings.calibration.camera_intrinsic.k,
                            ray_range, image_size, args.rosbag_path, ablation_name, idx)

    logdir = cloner_slam._log_directory

    if settings_description is not None:
        with open(f"{logdir}/configuration.txt", 'w+') as desc_file:
            desc_file.write(settings_description)

    for f in glob.glob("../outputs/frame*"):
        shutil.rmtree(f)

    bag = rosbag.Bag(args.rosbag_path, 'r')

    cloner_slam.start()

    start_time = None
    start_timestamp = None

    start_lidar_pose = None

    start_clock = None
    last_send = time.time()
    prev_timestamp = None

    for topic, msg, timestamp in bag.read_messages(topics=[lidar_topic, image_topic]):        
        # Wait for lidar to init
        if topic == lidar_topic and not init:
            init = True
            start_time = timestamp
        
        if not init:
            continue

        timestamp -= start_time
        
        if args.duration is not None and timestamp.to_sec() > args.duration:
            break

        if topic == image_topic:
            image = build_image_from_msg(msg, timestamp)

            try:
                lidar_tf = tf_buffer.lookup_transform_core('map', "lidar", timestamp + start_time)
            except tf2_py.ExtrapolationException as e:
                print("Skipping camera message: No valid TF")
                continue

            T_lidar = msg_to_transformation_mat(lidar_tf)

            if start_lidar_pose is None:
                start_lidar_pose = T_lidar

            gt_lidar_pose = start_lidar_pose.inverse() @ T_lidar
            gt_cam_pose = Pose(gt_lidar_pose @ lidar_to_camera)
            
            if prev_timestamp is not None:
                time.sleep(max(0,(timestamp - prev_timestamp).to_sec() - (time.time() - last_send)))
            cloner_slam.process_rgb(image, gt_cam_pose)
        elif topic == lidar_topic:
            lidar_scan = build_scan_from_msg(msg, timestamp)

            if prev_timestamp is not None:
                time.sleep(max(0,(timestamp - prev_timestamp).to_sec() - (time.time() - last_send)))

            cloner_slam.process_lidar(lidar_scan)
        else:
            raise Exception("Should be unreachable")
        last_send = time.time()
        prev_timestamp = timestamp

        if start_clock is None:
            start_clock = time.time()

    cloner_slam.stop()
    end_clock = time.time()


    with open(f"{cloner_slam._log_directory}/runtime.txt", 'w+') as runtime_f:
        runtime_f.write(f"Execution Time (With Overhead): {end_clock - init_clock}\n")
        runtime_f.write(f"Execution Time (Without Overhead): {end_clock - start_clock}\n")


    checkpoints = os.listdir(f"{logdir}/checkpoints")
    if len(checkpoints) == 0:
        return


    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
    checkpoint_path = pathlib.Path(f"{logdir}/checkpoints/{checkpoint}")
    ckpt = torch.load(str(checkpoint_path))

    kfs = ckpt["poses"]

    gt = []
    est = []

    for kf in kfs:
        if "gt_start_lidar_pose" in kf:
            gt_pose = Pose(pose_tensor = kf["gt_start_lidar_pose"])
            est_pose = Pose(pose_tensor = kf["start_lidar_pose"])
        else:
            gt_pose = Pose(pose_tensor=kf["gt_lidar_pose"])
            est_pose = Pose(pose_tensor=kf["lidar_pose"])
            
        gt.append(gt_pose.get_translation())
        est.append(est_pose.get_translation())

    gt = torch.stack(gt).detach().cpu()
    est = torch.stack(est).detach().cpu()

    diff = gt - est
    dist = torch.linalg.norm(diff, dim=1)
    rmse = torch.sqrt(torch.mean(dist**2))

    print(f"RMSE: {rmse:.3f}")

# Implements a single worker in a thread-pool model.
def _gpu_worker(args, gpu_id: int, job_queue: mp.Queue) -> None:

    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            return

        settings, description, trial_idx = data
        run_trial(args, settings, description, trial_idx)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run ClonerSLAM on RosBag")
    parser.add_argument("rosbag_path")
    parser.add_argument("calibration_path")
    parser.add_argument("experiment_name", nargs="?", default="experiment")
    parser.add_argument("--overrides", type=str, default=None, required=False)
    parser.add_argument("--duration", help="How long to run for (in input data time, sec)", type=float, default=None)
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = [0], help="Which GPUs to use. Defaults to parallel if set")
    parser.add_argument("--num_repeats", type=int, required=False, default=1, help="How many times to run the experiment")

    args = parser.parse_args()

    if args.overrides is not None:
        settings_options, settings_descriptions = \
            Settings.generate_options(os.path.expanduser("~/ClonerSLAM/cfg/default_settings.yaml"), 
                                      os.path.expanduser(args.overrides))
        
        settings_options = settings_options * args.num_repeats
        settings_descriptions = settings_descriptions * args.num_repeats
            
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")
        args.experiment_name = f"{args.experiment_name}_{now_str}"
        
    else:
        settings_descriptions = [None] * args.num_repeats
        settings_options = [Settings.load_from_file(os.path.expanduser("~/ClonerSLAM/cfg/default_settings.yaml"))] * args.num_repeats


    if args.gpu_ids != [0]:
        mp.set_start_method('spawn')

        job_queue_data = zip(settings_options, settings_descriptions, range(len(settings_descriptions)))

        job_queue = mp.Queue()
        for element in job_queue_data:
            job_queue.put(element)
        
        for _ in args.gpu_ids:
            job_queue.put(None)

        # Create the workers
        gpu_worker_processes = []
        for gpu_id in args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(args,gpu_id,job_queue,)))
            gpu_worker_processes[-1].start()

        # Sync
        for process in gpu_worker_processes:
            process.join()
        
                
    else:
        for idx, (settings, description) in enumerate(zip(settings_options, settings_descriptions)):
            if len(settings_options) == 1:
                idx = None
            run_trial(args, settings, description, idx)