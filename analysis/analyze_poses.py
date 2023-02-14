#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import pathlib
import pickle
import re
import sys
import time

import imageio
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
import torch
import tqdm

# autopep8: off
# Linting needs to be disabled here or it'll try to move includes before path.
PUB_ROS = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from src.common.pose import Pose


parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directories", nargs='+', type=str, help="folder in outputs with all results")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--title", type=str, default=None)

args = parser.parse_args()

for experiment_directory in args.experiment_directories:
    print(experiment_directory)
    checkpoints = os.listdir(f"{experiment_directory}/checkpoints")

    if args.ckpt_id is None:
        #https://stackoverflow.com/a/2669120
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
    else:
        checkpoint = f"ckpt_{args.ckpt_id}.tar"

    checkpoint_path = pathlib.Path(f"{experiment_directory}/checkpoints/{checkpoint}")

    if not checkpoint_path.exists():
        print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
        exit()
    ckpt = torch.load(str(checkpoint_path))

    kfs = ckpt["poses"]

    gt = []
    tracked = []
    est = []

    for kf in kfs:
        if "gt_start_lidar_pose" in kf:
            gt_pose = Pose(pose_tensor = kf["gt_start_lidar_pose"])
            est_pose = Pose(pose_tensor = kf["start_lidar_pose"])
            tracked_pose = Pose(pose_tensor=kf["tracked_start_lidar_pose"])
            use_simple_poses=False
        else:
            gt_pose = Pose(pose_tensor=kf["gt_lidar_pose"])
            est_pose = Pose(pose_tensor=kf["lidar_pose"])
            tracked_pose = Pose(pose_tensor=kf["tracked_pose"])
            use_simple_poses=True
            

        gt.append(gt_pose.get_translation())
        tracked.append(tracked_pose.get_translation())
        est.append(est_pose.get_translation())

    if use_simple_poses:
        pose_key = "lidar_pose"
    else:
        pose_key = "start_lidar_pose"
    translation_rel_errs = []
    for kf_a, kf_b in zip(kfs[:-1], kfs[1:]):
        est_start = Pose(pose_tensor = kf_a[pose_key])
        est_end = Pose(pose_tensor = kf_b[pose_key])
        
        gt_start = Pose(pose_tensor = kf_a["gt_" + pose_key])
        gt_end = Pose(pose_tensor = kf_b["gt_" + pose_key])

        est_delta = est_start.inv() * est_end
        gt_delta = gt_start.inv() * gt_end

        est_xyz = est_delta.get_translation()
        gt_xyz = gt_delta.get_translation()
        
        translation_rel_errs.append((gt_xyz - est_xyz).norm())

    translation_rel_errs = torch.tensor(translation_rel_errs)
    rmse_rel_err = torch.sqrt(torch.mean(translation_rel_errs**2))

    print("RMSE Relative Error:", rmse_rel_err)

    gt = torch.stack(gt).detach().cpu()
    tracked = torch.stack(tracked).detach().cpu()
    est = torch.stack(est).detach().cpu()

    tracked_rmse = torch.sqrt(torch.mean(np.square(torch.linalg.norm(tracked-gt, dim=1)))).item()
    est_rmse = torch.sqrt(torch.mean(torch.square(torch.linalg.norm(est-gt, dim=1)))).item()



    ax = plt.gca()
    ax.set_aspect('equal')

    plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
    plt.plot(est[:,0], est[:,1], label="Optimized")
    plt.scatter(gt[0,0],gt[0,1], s=20, color='red', label="Start Point")
    # plt.plot(tracked[:,0], tracked[:,1], color='g',label="Tracked", linestyle="dashed")

    text_box = AnchoredText(f"KF Relative RMSE: {rmse_rel_err:.3f}\nOptimized RMSE:{est_rmse:.3f}", 
                            frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ax.add_artist(text_box)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    if args.title is not None:
        plt.title(args.title)

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{experiment_directory}/poses.png")
    plt.clf()
