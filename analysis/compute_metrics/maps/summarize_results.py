import argparse
import yaml
import glob
import numpy as np

#MODIFY These!
configs = [f"config_{i}" for i in range(3)]
config_names = configs
datasets = ["canteen", "garden", "mcr", "quad"]
metrics = ["l1"]
# metrics = ["accuracy", "completion", "precision", "recall"]


parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str)

args = parser.parse_args()




# results [seq, alg, stat] = [val]num_trials
results = {}



for seq in datasets:
  results[seq] = {}
  for alg in configs:
    results[seq][alg] = {}
    for fname in sorted(glob.glob(f"{args.input_path}/{seq}/{alg}/stat*")):
      with open(fname, 'r') as f:
        stats = yaml.full_load(f)
      
      for stat in metrics:
        if stat == "l1": continue

        if stat not in results[seq][alg]:
          results[seq][alg][stat] = []
        results[seq][alg][stat].append(float(stats[stat]))

    for fname in sorted(glob.glob(f"{args.input_path}/{seq}/{alg}/l1*")):
      with open(fname, 'r') as f:
        l1stats = yaml.full_load(f)
        
        if "l1" not in results[seq][alg]:
          results[seq][alg]["l1"] = []
        results[seq][alg]["l1"].append(l1stats["mean"])

metrics_str = "&".join(metrics)
min_table = "&" + f" {metrics_str} ".join(datasets) + "\\\\\n"
median_table= "&" + f" {metrics_str} ".join(datasets) + "\\\\\n"
mean_table = "&" + f" {metrics_str} ".join(datasets) + "\\\\\n"

median_csv = "," + (","*len(metrics)).join(datasets) + ","*len(metrics) + "\n"

for alg, algname in zip(configs, config_names):
  median_table += algname
  mean_table += algname
  min_table += algname

  median_csv += algname

  for seq in datasets:
    for metric in metrics:
      entries = np.array(results[seq][alg][metric])

      min_table += f"&{entries.min():.3f}  "
      median_table += f"&{np.median(entries):.3f}  "
      mean_table += f"&${np.mean(entries):.3f}\\pm{np.std(entries):.3f}$  "
      median_csv += f",{np.median(entries)}"

  median_table += "\\\\\n"
  mean_table += "\\\\\n"
  min_table += "\\\\\n"
  median_csv += "\n"


with open(f"./{args.input_path}/table_median.tex", 'w+') as f:
    f.write(median_table)

with open(f"./{args.input_path}/table_mean.tex", 'w+') as f:
    f.write(mean_table)
      
with open(f"./{args.input_path}/table_min.csv", 'w+') as f:
    f.write(min_table)

with open(f"./{args.input_path}/table_median.csv", 'w+') as f:
    f.write(median_csv)
