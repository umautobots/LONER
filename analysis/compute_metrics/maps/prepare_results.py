import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str)
parser.add_argument("output_dir")
parser.add_argument("dataset")
parser.add_argument("--single_config", action="store_true", default=False)
parser.add_argument("--single_trial", action="store_true", default=False)
args = parser.parse_args()

experiment_dir = args.experiment_dir
dataset = args.dataset.lower()

output_dir = f"{args.output_dir}/{dataset}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

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
        os.makedirs(config_out_dir, exist_ok=True)

    if not args.single_trial:
        trials = os.listdir(f"{args.experiment_dir}/{config}")
    else:
        trials = [""]

    trials = [t for t in trials if os.path.isdir(f"{config_dir}/{t}")]

    for trial_idx, trial in enumerate(trials):
        trial_dir = f"{config_dir}/{trial}"

        if os.path.exists(f"{trial_dir}/metrics/statistics.yaml"):
            shutil.copy(f"{trial_dir}/metrics/statistics.yaml", f"{config_out_dir}/statistics_{trial_idx}.yaml")
        if os.path.exists(f"{trial_dir}/metrics/l1.yaml"):  
            shutil.copy(f"{trial_dir}/metrics/l1.yaml", f"{config_out_dir}/l1_{trial_idx}.yaml")
