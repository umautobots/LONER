#!/usr/bin/env python
# coding: utf-8

"""
File: prepare_for_traj_eval.py
"""


import argparse
import os
import pathlib
import pickle
import sys
import pandas as pd
import yaml

import numpy as np

PLATFORM = "desktop"


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose import Pose


def prepare_sequence(trajectory_path, output_path):
    est_data = pd.read_csv(trajectory_path, header=None, delimiter=" ")
    est_data[0] = est_data[0] - est_data[0][0]

    os.makedirs(pathlib.Path(output_path).parent.as_posix(), exist_ok=True)

    np.savetxt(output_path, est_data.to_numpy(), delimiter=" ", fmt="%.10f")


def write_config_data(path, gt_data):
    eval_cfg = {"align_type": "se3", "align_num_frames": -1}

    with open(f"{path}/eval_cfg.yaml", 'w') as f:
        yaml.dump(eval_cfg, f)

    np.savetxt(f"{path}/stamped_groundtruth.txt", gt_data, delimiter=" ", fmt="%.10f")


parser = argparse.ArgumentParser(description="Analyze KeyFrame Poses")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
parser.add_argument("sequence_name", type=str, help="name of sequence")
parser.add_argument("--study_name", type=str, required=False, default="cloner_slam_evaluation")

args = parser.parse_args()

multiple_configs = False
multiple_trials = False

for item in os.listdir(args.experiment_directory):
    item_path = os.path.join(args.experiment_directory, item)
    if os.path.isdir(item_path) and item[:6] == "config":
        multiple_configs = True
        for subitem in os.listdir(os.path.join(args.experiment_directory, item)):
            if os.path.isdir(f"{item_path}/{subitem}") and subitem[:5] == "trial":
                multiple_trials = True
                break
        break
    if os.path.isdir(item_path) and item[:5] == "trial":
        multiple_trials = True
        break


rpg_output_dir = f"{args.experiment_directory}/rpg_evaluation/{args.study_name}"

os.makedirs(f"{rpg_output_dir}/{PLATFORM}", exist_ok=True)

trial_names = None
config_names = [None]
if multiple_configs:
    config_names = [i for i in os.listdir(args.experiment_directory) if i[:6] == "config"]
    if multiple_trials:
        trial_names = [i for i in os.listdir(f"{args.experiment_directory}/{config_names[0]}") if i[:5] == "trial"]
elif multiple_trials:
    trial_names = [i for i in os.listdir(f"{args.experiment_directory}") if i[:5] == "trial"]


pkl_path = args.experiment_directory
if multiple_configs:
    pkl_path += f"/{config_names[0]}/"
if multiple_trials:
    pkl_path += f"/{trial_names[0]}/"

with open(f"{pkl_path}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

gt_path = os.path.expanduser(full_config["run_config"]["groundtruth_traj"])
gt_data = pd.read_csv(gt_path, header=None, delimiter=" ")
gt_data[0] = gt_data[0] - gt_data[0][0]


for config in config_names:
    if config is None:
        config = "cloner_slam"
        config_path = args.experiment_directory
    else:
        config_path = f"{args.experiment_directory}/{config}"


    config_output_path = f"{rpg_output_dir}/{PLATFORM}/{config}/{PLATFORM}_{config}_{args.sequence_name}"

    os.makedirs(config_output_path, exist_ok=True)

    write_config_data(config_output_path, gt_data)

    if multiple_trials:
        for trial_idx, trial in enumerate(trial_names):
            traj_path = f"{config_path}/{trial}/trajectory/estimated_trajectory.txt"
            output_path = f"{config_output_path}/stamped_traj_estimate{trial_idx}.txt"
            prepare_sequence(traj_path, output_path)
    else:
        traj_path = f"{config_path}/trajectory/estimated_trajectory.txt"
        output_path = f"{config_output_path}/stamped_traj_estimate.txt"
        prepare_sequence(traj_path, output_path)

if not multiple_configs:
    config_names = ["cloner_slam"]

analyze_config = {
    "Datasets": {
        args.sequence_name: {
            "label": args.sequence_name
        }
    },
    "Algorithms": {c: {"fn": "traj_est", "label": c} for c in config_names},
    "RelDistances": [1,10,25]
}

with open(f"{rpg_output_dir}/{args.sequence_name}_evaluation.yaml", 'w+') as f:
    yaml.dump(analyze_config, f)