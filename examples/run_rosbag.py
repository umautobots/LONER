#!/usr/bin/env python3

import argparse
import datetime
import os
import sys
import re
import pathlib
import time

import cv2
import pandas as pd
import yaml
import ros_numpy
import rosbag
import rospy
import tf2_py
import torch
from attrdict import AttrDict
from cv_bridge import CvBridge
from pathlib import Path
from sensor_msgs.msg import Image, PointCloud2
import torch.multiprocessing as mp

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from src.loner import Loner
from src.common.pose import Pose
from src.common.sensors import Image, LidarScan
from src.common.settings import Settings

from utils import *

LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf

bridge = CvBridge()

WARN_MOCOMP_ONCE = True
WARN_LIDAR_TIMES_ONCE = True

def build_scan_from_msg(lidar_msg: PointCloud2, timestamp: rospy.Time) -> LidarScan:

    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(
        lidar_msg)

    lidar_data = torch.from_numpy(pd.DataFrame(lidar_data).to_numpy()).cuda()
    mask = torch.ones((lidar_data.shape[0]))
    xyz = lidar_data[:, :3]
    
    dists = torch.linalg.norm(xyz, dim=1)
    valid_ranges = dists > LIDAR_MIN_RANGE

    xyz = xyz[valid_ranges].T

    fields = [f.name for f in lidar_msg.fields]
    time_idx = None
    for f_idx, f in enumerate(fields):
        if "time" in f:
            time_idx = f_idx
            break

    
    global WARN_MOCOMP_ONCE

    if time_idx is None:
        if WARN_MOCOMP_ONCE:
            print("Warning: LiDAR Data has No Associated Timestamps. Motion compensation is useless.")
            WARN_MOCOMP_ONCE = False
        timestamps = torch.full_like(xyz[0], timestamp.to_sec()).float()
    else:

        timestamps = lidar_data[valid_ranges, time_idx]

        global WARN_LIDAR_TIMES_ONCE
        if timestamps[0] < 1e-5:
            if WARN_LIDAR_TIMES_ONCE:
                print("Assuming LiDAR timestamps within a scan are local, and start at 0")
            timestamps += timestamp.to_sec()
        else:
            if WARN_LIDAR_TIMES_ONCE:
                print("Assuming lidar timestamps within a scan are global.")
            timestamps = timestamps - timestamps[0] + timestamp.to_sec()
        WARN_LIDAR_TIMES_ONCE = False


        if timestamps[-1] - timestamps[0] < 1e-3:
            if WARN_MOCOMP_ONCE:
                print("Warning: Timestamps in LiDAR data aren't unique. Motion compensation is useless")
                WARN_MOCOMP_ONCE = False

            timestamps = torch.full_like(xyz[0], timestamp.to_sec()).float()

    timestamps = timestamps.float()

    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:, indices]

    mask = mask[valid_ranges].float()
    mask = mask[indices]

    return LidarScan(directions.float().cpu(), dists.float().cpu(), timestamps.float().cpu(), mask.float().cpu())


def tf_to_settings(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    xyz = [trans.x, trans.y, trans.z]
    quat = [rot.w, rot.x, rot.y, rot.z]

    return AttrDict({"xyz": xyz, "orientation": quat})


def build_image_from_msg(image_msg, timestamp, scale_factor) -> Image:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_img = cv2.resize(cv_img, (0,0), fx=scale_factor, fy=scale_factor)
    pytorch_img = torch.from_numpy(cv_img / 255).float()
    return Image(pytorch_img, timestamp.to_sec())


def run_trial(config, settings, settings_description = None, config_idx = None, trial_idx = None):
    im_scale_factor = settings.system.image_scale_factor

    rosbag_path = Path(os.path.expanduser(config["dataset"]))
    
    init = False

    init_clock = time.time()
    
    calibration = load_calibration(config["dataset_family"], config["calibration"])

    camera = settings.system.ros_names.camera
    lidar = settings.system.ros_names.lidar
    camera_suffix = settings.system.ros_names.camera_suffix
    topic_prefix = settings.system.ros_names.topic_prefix

    lidar_topic = f"{topic_prefix}/{lidar}"

    lidar_only = settings.system.lidar_only

    if not lidar_only:
        image_topic = f"{topic_prefix}/{camera}/{camera_suffix}"
        
        K = torch.from_numpy(calibration.left_cam_intrinsic["K"]).float()
        K[:2, :] *= im_scale_factor
        settings["calibration"]["camera_intrinsic"]["k"] = K

        new_k = torch.from_numpy(calibration.left_cam_intrinsic["projection_matrix"]).float()[:3,:3]
        new_k[:2, :] *= im_scale_factor
        settings["calibration"]["camera_intrinsic"]["new_k"] = new_k

        settings["calibration"]["camera_intrinsic"]["distortion"] = \
            torch.from_numpy(calibration.left_cam_intrinsic["distortion_coeffs"]).float()

        settings["calibration"]["camera_intrinsic"]["width"] = int(calibration.left_cam_intrinsic["width"] // (1/im_scale_factor))
        
        settings["calibration"]["camera_intrinsic"]["height"] = int(calibration.left_cam_intrinsic["height"] // (1/im_scale_factor))

        image_size = (settings.calibration.camera_intrinsic.height, settings.calibration.camera_intrinsic.width)

        lidar_to_camera = Pose.from_settings(calibration.t_lidar_to_left_cam).get_transformation_matrix()
        camera_to_lidar = lidar_to_camera.inverse()

        settings["calibration"]["lidar_to_camera"] = calibration.t_lidar_to_left_cam
    else:
        camera_to_lidar = None
        image_size = None

    ray_range = settings.mapper.optimizer.model_config.data.ray_range

    settings["experiment_name"] = config["experiment_name"]

    settings["run_config"] = config

    loner = Loner(settings)

    # Get ground truth trajectory. This is only used to construct the world cube.
    if config["groundtruth_traj"] is not None:
        ground_truth_file = os.path.expanduser(config["groundtruth_traj"])
        ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
        lidar_poses, timestamps = build_poses_from_df(ground_truth_df)
        tf_buffer, timestamps = build_buffer_from_poses(lidar_poses, timestamps)
    else:
        tf_buffer = None
        lidar_poses = None

    if config_idx is None and trial_idx is None:
        ablation_name = None
    else:
        ablation_name = config["experiment_name"]

    if settings.system.world_cube.compute_from_groundtruth:
        assert lidar_poses is not None, "Must provide groundtruth file, or set system.world_cube.compute_from_groundtruth=False"
        traj_bounding_box = None
        lidar_poses_init = lidar_poses
    else:
        lidar_poses_init = None
        traj_bounding_box = settings.system.world_cube.trajectory_bounding_box

    loner.initialize(camera_to_lidar, lidar_poses_init, settings.calibration.camera_intrinsic.k,
                            ray_range, image_size, rosbag_path.as_posix(), ablation_name, config_idx, trial_idx,
                            traj_bounding_box)

    logdir = loner._log_directory

    if settings_description is not None:
        if trial_idx == 0:
            with open(f"{logdir}/../configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
        elif trial_idx is None:
            with open(f"{logdir}/configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
    

    bag = rosbag.Bag(rosbag_path.as_posix(), 'r')

    loner.start()

    start_time = None

    start_lidar_pose = None

    start_clock = None

    warned_skip_once = False

    topics = [lidar_topic] if lidar_only else [lidar_topic, image_topic]
    
    for topic, msg, timestamp in bag.read_messages(topics=topics):        
        # Wait for lidar to init
        if topic == lidar_topic and (not init) and timestamp.to_sec() and (tf_buffer is None or timestamp >= timestamps[0]):
            init = True
            start_time = timestamp
        
        if not init:
            continue
        
        
        timestamp -= start_time
        
        if config["duration"] is not None and timestamp.to_sec() > config["duration"]:
            break

        if (not lidar_only) and topic == image_topic:
            image = build_image_from_msg(msg, timestamp, im_scale_factor)
            loner.process_rgb(image)
        elif topic == lidar_topic:
            if tf_buffer is not None:
                try:
                    lidar_tf = tf_buffer.lookup_transform_core('map', "lidar", timestamp + start_time)
                except tf2_py.ExtrapolationException as e:
                    if not warned_skip_once:
                        print("Warning: Skipping a camera message: No valid TF.")
                        warned_skip_once = True
                    continue

                T_lidar = msg_to_transformation_mat(lidar_tf)

                if start_lidar_pose is None:
                    start_lidar_pose = T_lidar

                gt_lidar_pose = start_lidar_pose.inverse() @ T_lidar
            else:
                gt_lidar_pose = torch.eye(4)

            lidar_scan = build_scan_from_msg(msg, timestamp)

            loner.process_lidar(lidar_scan, Pose(gt_lidar_pose))

        else:
            raise Exception("Should be unreachable")

        if start_clock is None:
            start_clock = time.time()

    loner.stop()
    end_clock = time.time()


    with open(f"{loner._log_directory}/runtime.txt", 'w+') as runtime_f:
        runtime_f.write(f"Execution Time (With Overhead): {end_clock - init_clock}\n")
        runtime_f.write(f"Execution Time (Without Overhead): {end_clock - start_clock}\n")


    checkpoints = os.listdir(f"{logdir}/checkpoints")
    if len(checkpoints) == 0:
        return

    ### Compute RMSE trajectory error

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
        gt_pose = Pose(pose_tensor=kf["gt_lidar_pose"])
        est_pose = Pose(pose_tensor=kf["lidar_pose"])
            
        gt.append(gt_pose.get_translation())
        est.append(est_pose.get_translation())

    gt = torch.stack(gt).detach().cpu()
    est = torch.stack(est).detach().cpu()

    diff = gt - est
    dist = torch.linalg.norm(diff, dim=1)
    rmse = torch.sqrt(torch.mean(dist**2))

    print(f"Unaligned KeyFrame RMSE: {rmse:.3f}")


# Implements a single worker in a thread-pool model.
def _gpu_worker(config, gpu_id: int, job_queue: mp.Queue) -> None:

    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            return

        settings, description, config_idx, trial_idx = data
        run_trial(config, settings, description, config_idx, trial_idx)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Loner SLAM on RosBag")
    parser.add_argument("configuration_path")
    parser.add_argument("experiment_name", nargs="?", default=None)
    parser.add_argument("--duration", help="How long to run for (in input data time, sec)", type=float, default=None)
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = [0], help="Which GPUs to use. Defaults to parallel if set")
    parser.add_argument("--num_repeats", type=int, required=False, default=1, help="How many times to run the experiment")
    parser.add_argument("--run_all_combos", action="store_true",default=False, help="If set, all combinations of overrides will be run. Otherwise, one changed at a time.")
    parser.add_argument("--overrides", type=str, default=None, help="File specifying parameters to vary for ablation study or testing")
    parser.add_argument("--lite", action="store_true",default=False, help="If set, uses the lite model configuration instead of the full model.")

    args = parser.parse_args()


    with open(args.configuration_path) as config_file:
        config = yaml.full_load(config_file)

    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    config["duration"] = args.duration

    if args.lite:
        lite_mode_path = os.path.expanduser("~/LonerSLAM/cfg/loner_slam_lite.yaml")
        with open(lite_mode_path, 'r') as lite_mode_f:
            lite_mode_changes = yaml.full_load(lite_mode_f)
    else:
        lite_mode_changes = None

    baseline_settings_path = os.path.expanduser(f"~/LonerSLAM/cfg/{config['baseline']}")

    if args.overrides is not None:
        settings_options, settings_descriptions = \
            Settings.generate_options(baseline_settings_path,
                                      args.overrides,
                                      args.run_all_combos,
                                      [config["changes"], lite_mode_changes])
        
        settings_options = settings_options
        settings_descriptions = settings_descriptions
    else:
        settings_descriptions = [None]
        settings_options = [Settings.load_from_file(baseline_settings_path)]            

        if config["changes"] is not None:
                settings_options[0].augment(config["changes"])
                
        if args.lite:
            settings_options[0].augment(lite_mode_changes)

    if args.overrides is not None or args.num_repeats > 1:
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")
        config["experiment_name"] += f"_{now_str}"

    if len(args.gpu_ids) > 1:
        mp.set_start_method('spawn')

        job_queue_data = zip(settings_options, settings_descriptions, range(len(settings_descriptions)))

        job_queue = mp.Queue()
        for element in job_queue_data:
            if args.num_repeats == 1:
                job_queue.put(element + (None,))
            else:
                for trial_idx in range(args.num_repeats):
                    job_queue.put(element + (trial_idx,))
        
        for _ in args.gpu_ids:
            job_queue.put(None)

        # Create the workers
        gpu_worker_processes = []
        for gpu_id in args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(config,gpu_id,job_queue,)))
            gpu_worker_processes[-1].start()

        # Sync
        for process in gpu_worker_processes:
            process.join()
        
                
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
        for config_idx, (settings, description) in enumerate(zip(settings_options, settings_descriptions)):
            if len(settings_options) == 1:
                config_idx = None
            for trial_idx in range(args.num_repeats):
                if args.num_repeats == 1:
                    trial_idx = None
                run_trial(config, settings, description, config_idx, trial_idx)