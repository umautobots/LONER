import argparse
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd


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

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")

args = parser.parse_args()

keyframe_folders = sort_alphanum(os.listdir(f"{args.experiment_directory}/losses"))

with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    full_config = pickle.load(f)

keyframe_schedule = full_config.mapper.optimizer.keyframe_schedule


os.makedirs(f"{args.experiment_directory}/loss_plots", exist_ok=True)


fig, ax = plt.subplots(2, 3)
for kf_idx, keyframe in enumerate(keyframe_folders):

    if kf_idx < 100:
        continue

    phases = sort_alphanum(os.listdir(f"{args.experiment_directory}/losses/{keyframe}"))
     

    for phase_idx, phase_file in enumerate(phases):
        fname = f"{args.experiment_directory}/losses/{keyframe}/{phase_file}"
        
        data = pd.read_csv(fname,names=["L"])["L"].to_numpy()
        if phase_idx == 0:
            continue

        col = (kf_idx-100) % 3
        row = (kf_idx-100) // 3

        ax[row][col].plot(data)
        ax[row][col].set_xlabel("Iteration")
        ax[row][col].set_ylabel("Loss")
        ax[row][col].set_title(f"KeyFrame {kf_idx}")
        os.makedirs(f"{args.experiment_directory}/loss_plots/keyframe_{kf_idx}", exist_ok=True)
        
    if kf_idx == 105:
        break

plt.tight_layout()
plt.savefig(f"{args.experiment_directory}/loss_plots/late_8.png")
plt.clf()
    
            