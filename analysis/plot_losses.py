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
import pandas as pd
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

def sort_alphanum(input):
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(input, key = alphanum_key)

def save_logs(kf_schedule_logs, schedule_idx, super_verbose=False):
    for phase_idx, phase in enumerate(kf_schedule_logs):
        
        if super_verbose:
            trials = phase
        else:
            phase = np.vstack(phase)
            mean_loss = np.mean(phase, axis=0)
            trials = [mean_loss]
        
        for trial_idx, trial in enumerate(trials):
            plt.plot(trial)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Stage {schedule_idx}, Phase {phase_idx}")
            plt.savefig(f"{args.experiment_directory}/loss_plots/stage_{schedule_idx}_phase_{phase_idx}_trial_{trial_idx}")
            plt.clf()

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

args = parser.parse_args()

keyframe_folders = sort_alphanum(os.listdir(f"{args.experiment_directory}/losses"))

with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

keyframe_schedule = full_config.mapper.optimizer.keyframe_schedule


os.makedirs(f"{args.experiment_directory}/loss_plots", exist_ok=True)



prev_kf_schedule_idx = -1
current_kf_schedule_logs = []
for kf_idx, keyframe in enumerate(keyframe_folders):

    cumulative_kf_idx = 0
    for kf_schedule_idx, item in enumerate(keyframe_schedule):
        kf_count = item["num_keyframes"]
        iteration_schedule = item["iteration_schedule"]

        cumulative_kf_idx += kf_count
        if cumulative_kf_idx >= kf_idx + 1 or kf_count == -1:
            break

    phases = sort_alphanum(os.listdir(f"{args.experiment_directory}/losses/{keyframe}"))
     
    if prev_kf_schedule_idx != kf_schedule_idx:

        if len(current_kf_schedule_logs) > 0:
            save_logs(current_kf_schedule_logs, prev_kf_schedule_idx)

        current_kf_schedule_logs = [[] for _ in range(len(phases))]    

    for phase_idx, phase_file in enumerate(phases):
        fname = f"{args.experiment_directory}/losses/{keyframe}/{phase_file}"
        
        data = pd.read_csv(fname,names=["L"])["L"].to_numpy()
        current_kf_schedule_logs[phase_idx].append(data)

    prev_kf_schedule_idx = kf_schedule_idx

save_logs(current_kf_schedule_logs, prev_kf_schedule_idx, True)
