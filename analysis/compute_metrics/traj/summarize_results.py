import argparse
import pandas as pd

## Modify These!
configs = [f"config_{i}" for i in range(3)]
config_names = configs
datasets = ["canteen", "garden", "mcr", "quad"]
metrics = ["ape"]


parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str)

args = parser.parse_args()

num_estimates=5

results = {}

for metric in metrics:
    results[metric] = {}
    for seq in datasets:
        results[metric][seq] = {}
        for alg in configs:
            data_path = f"{args.input_path}/{seq}/results/{alg}/statistics_{metric}.csv"
            data = pd.read_csv(data_path)
            mean_out = data["rmse"].mean()
            std_out = data["rmse"].std()

            median_out = data["rmse"].median()
            min_out = data["rmse"].min()

            results[metric][seq][alg] = (mean_out, std_out, median_out, min_out)

min_table = "&" + " &&& ".join(datasets) + "\\\\\n"
median_table = "&" + " &&& ".join(datasets) + "\\\\\n"
mean_table = "&" + " &&& ".join(datasets) + "\\\\\n"
mean_csv = "Alg," + ",".join(metrics) + "\n"
median_csv = "," + (","*len(metrics)).join(datasets) + ","*len(metrics) + "\n"

for alg, algname in zip(configs, config_names):
    median_table += algname
    mean_table += algname
    mean_csv += algname
    min_table += algname
    median_csv += algname
    for seq in datasets:
        for metric in metrics:
            min_table += f"&{results[metric][seq][alg][3]:.3f}  "
            median_table += f"&{results[metric][seq][alg][2]:.3f}  "
            mean_table += f"&${results[metric][seq][alg][0]:.3f}\\pm{results[metric][seq][alg][1]:.3f}$  "
            mean_csv += f",{results[metric][seq][alg][0]}"
            median_csv += f",{results[metric][seq][alg][2]}"

    median_table += "\\\\\n"
    mean_table += "\\\\\n"
    min_table += "\\\\\n"
    mean_csv += "\n"
    median_csv += "\n"

with open(f"./{args.input_path}/table_median.tex", 'w+') as f:
    f.write(median_table)

with open(f"./{args.input_path}/table_mean.tex", 'w+') as f:
    f.write(mean_table)
        
with open(f"./{args.input_path}/table_median.csv", 'w+') as f:
    f.write(median_csv)

with open(f"./{args.input_path}/table_min.csv", 'w+') as f:
    f.write(min_table)
