#!/usr/bin/env python
# coding: utf-8

import argparse
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
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--title", type=str, default=None)

args = parser.parse_args()

checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

if args.ckpt_id is None:
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"

checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")

if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()
ckpt = torch.load(str(checkpoint_path))

kfs = ckpt["poses"]

gt = []
tracked = []
est = []

for kf in kfs:
    tracked_pose = Pose(pose_tensor = kf["tracked_start_lidar_pose"])
    gt_pose = Pose(pose_tensor = kf["gt_start_lidar_pose"])
    est_pose = Pose(pose_tensor = kf["start_lidar_pose"])

    gt.append(gt_pose.get_translation())
    tracked.append(tracked_pose.get_translation())
    est.append(est_pose.get_translation())


gt = torch.stack(gt).detach().cpu()
tracked = torch.stack(tracked).detach().cpu()
est = torch.stack(est).detach().cpu()

tracked_rmse = torch.sqrt(torch.mean(np.square(tracked-gt))).item()
est_rmse = torch.sqrt(torch.mean(torch.square(est-gt))).item()


ax = plt.gca()
ax.set_aspect('equal')

plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
plt.plot(est[:,0], est[:,1], label="Optimized")
plt.plot(tracked[:,0], tracked[:,1], label="Tracked", linestyle="dashed")

text_box = AnchoredText(f"Tracked RMSE: {tracked_rmse:.3f}\nOptimized RMSE:{est_rmse:.3f}", 
                         frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
ax.add_artist(text_box)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

if args.title is not None:
    plt.title(args.title)

plt.legend()

plt.savefig(f"{args.experiment_directory}/poses.png")