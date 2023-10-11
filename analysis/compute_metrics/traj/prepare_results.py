import os
import argparse
import shutil
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str, help="Path where the outputs from the experiment are stored")
parser.add_argument("output_dir", type=str, help="Location to store the output files")
parser.add_argument("dataset", type=str, help="A name string identifying the dataset processed in this experiment")
parser.add_argument("groundtruth_traj_path", type=str, help="A path to the groundtruth trajectory in TUM format.")
parser.add_argument("--single_config", action="store_true", default=False)
parser.add_argument("--single_trial", action="store_true", default=False)
args = parser.parse_args()

experiment_dir = args.experiment_dir
dataset = args.dataset.lower()

output_dir = f"{args.output_dir}/{dataset}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not args.single_config:
    configs = os.listdir(experiment_dir)
else: 
    configs = [""]

for config in configs:
    config_dir = f"{experiment_dir}/{config}"

    if not os.path.isdir(config_dir):
        continue

    config_out_dir = f"{output_dir}/{config}"

    if not args.single_config and not os.path.exists(config_out_dir):
        os.makedirs(config_out_dir)

    if not args.single_trial:
        trials = os.listdir(f"{args.experiment_dir}/{config}")
    else:
        trials = [""]

    trials = [t for t in trials if os.path.isdir(f"{config_dir}/{t}")]

    for trial_idx, trial in enumerate(trials):
        trial_dir = f"{config_dir}/{trial}"
        
        try:
            shutil.copy(f"{trial_dir}/trajectory/estimated_trajectory.txt", f"{config_out_dir}/stamped_traj_estimate{trial_idx}.txt")
        except Exception:
            print(f"Can't find: {trial_dir}/trajectory/estimated_trajectory.txt. Skipping")

gt_df = pd.read_csv(args.groundtruth_traj_path, delimiter=' ', header=None)
gt_df[0] = gt_df[0] - gt_df[0][0] # start times at 0
gt_data = gt_df.to_numpy()

with open(f"{output_dir}/stamped_groundtruth.txt", 'w+') as out_file:
    np.savetxt(out_file, gt_data, delimiter=" ", fmt="%10.10f")
