#!/usr/bin/env python
# coding: utf-8

"""
File: analyze_poses.py
Description: Reads in the optimized keyframe poses from the checkpoint, plots them, and computes metrics.

This is just a rough evaluation - full evaluations are conducted using other third-party packages.
See the README for more info.
"""


import argparse
import os
import pathlib
import re
import sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose import Pose

parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
parser.add_argument("experiment_directories", nargs='+', type=str, help="folder in outputs with all results")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--plot_tracked", action='store_true', default=False)

args = parser.parse_args()


all_gts = []
all_ests = []
all_relatives = []
all_rmses = []

convert = lambda text: int(text) if text.isdigit() else text 
alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
experiment_directories = sorted(args.experiment_directories, key = alphanum_key)

for experiment_directory in experiment_directories:
    checkpoints = os.listdir(f"{experiment_directory}/checkpoints")
    if args.ckpt_id is None:
        if "final.tar" in checkpoints:
            checkpoint = "final.tar"
        else:
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
    print("Loading checkpoint from", checkpoint_path.as_posix())
    ckpt = torch.load(str(checkpoint_path))

    kfs = ckpt["poses"]

    gt = []
    tracked = []
    est = []

    for kf in kfs:

        gt_pose = Pose(pose_tensor=kf["gt_lidar_pose"])
        est_pose = Pose(pose_tensor=kf["lidar_pose"])
        tracked_pose = Pose(pose_tensor=kf["tracked_pose"])            

        gt.append(gt_pose.get_translation())
        tracked.append(tracked_pose.get_translation())
        est.append(est_pose.get_translation())

    translation_rel_errs = []
    rel_errs_tracked = []
    for kf_a, kf_b in zip(kfs[:-1], kfs[1:]):
        est_start = Pose(pose_tensor = kf_a["lidar_pose"])
        est_end = Pose(pose_tensor = kf_b["lidar_pose"])
        
        gt_start = Pose(pose_tensor = kf_a["gt_lidar_pose"])
        gt_end = Pose(pose_tensor = kf_b["gt_lidar_pose"])
        
        est_delta = est_start.inv() * est_end
        gt_delta = gt_start.inv() * gt_end

        est_xyz = est_delta.get_translation()
        gt_xyz = gt_delta.get_translation()
        
        tracked_start = Pose(pose_tensor=kf_a["tracked_pose"])
        tracked_end = Pose(pose_tensor=kf_b["tracked_pose"])
        tracked_delta = tracked_start.inv() * tracked_end
        tracked_xyz = tracked_delta.get_translation()

        translation_rel_errs.append((gt_xyz - est_xyz).norm())
        rel_errs_tracked.append((gt_xyz - tracked_xyz).norm())

    translation_rel_errs = torch.tensor(translation_rel_errs)
    rmse_rel_err = torch.sqrt(torch.mean(translation_rel_errs**2))

    rel_errs_tracked = torch.tensor(rel_errs_tracked)
    rmse_rel_err_tracked = torch.sqrt(torch.mean(rel_errs_tracked**2))
    
    if os.path.exists(f"{experiment_directory}/configuration.txt"):
        with open(f"{experiment_directory}/configuration.txt") as f:
            cfg = f.read()
            print(f"\n\n===Configuration:===\n{cfg}")

    print("RMSE Relative Error:", rmse_rel_err)

    gt = torch.stack(gt).detach().cpu()
    tracked = torch.stack(tracked).detach().cpu()
    est = torch.stack(est).detach().cpu()

    tracked_rmse = torch.sqrt(torch.mean(torch.square(torch.linalg.norm(tracked-gt, dim=1)))).item()
    est_rmse = torch.sqrt(torch.mean(torch.square(torch.linalg.norm(est-gt, dim=1)))).item()
    
    print("Est RMSE: ", est_rmse)
    all_relatives.append(rmse_rel_err)
    all_rmses.append(est_rmse)

    if args.plot_tracked:
        fig, ax = plt.subplots(1,2)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')


        ax[0].plot(gt[:,0], gt[:,1], label="Ground Truth")
        ax[1].plot(gt[:,0], gt[:,1], label="Ground Truth")
        
        ax[0].plot(tracked[:,0], tracked[:,1], label="Tracked")
        ax[1].plot(est[:,0], est[:,1], label="Optimized")
        
        
        ax[0].scatter(gt[0,0],gt[0,1], s=20, color='red', label="Start Point")
        ax[1].scatter(gt[0,0],gt[0,1], s=20, color='red', label="Start Point")

        text_box = AnchoredText(f"RMSE Rel Err: {rmse_rel_err_tracked:.3f}\nRMSE ATE:{tracked_rmse:.3f}", 
                        frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax[0].add_artist(text_box)


        text_box = AnchoredText(f"RMSE Rel Err: {rmse_rel_err:.3f}\nRMSE ATE:{est_rmse:.3f}", 
                                frameon=True, loc=4, pad=0.5)

        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax[1].add_artist(text_box)

        ax[0].set_xlabel("X (m)")
        ax[1].set_xlabel("X (m)")
        ax[0].set_ylabel("Y (m)")
        ax[1].set_ylabel("Y (m)")
        
        ax[0].set_title("ICP Tracking Performance")
        ax[1].set_title("Optimized Performance")
        if args.title is not None:
            plt.suptitle(args.title)

        ax[0].legend(bbox_to_anchor=(1., 1.05))

        plt.savefig(f"{experiment_directory}/poses.png", dpi=1000)
        plt.show()
        plt.clf()
    else:
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
        plt.plot(est[:,0], est[:,1], label="Optimized")
        plt.scatter(gt[0,0],gt[0,1], s=20, color='red', label="Start Point")

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
        # plt.show()
        plt.clf()
    
    all_gts.append(gt)
    all_ests.append(est)