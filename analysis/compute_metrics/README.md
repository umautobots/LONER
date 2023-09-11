# Computing Metrics

This document outlines how to compute the metrics we report in our paper, split into trajectory metrics and mapping metrics. We use a combination of external libraries and custom scripts to handle this.

The process for this is a bit convoluted. Feel free to open issues or email me at sethgi \[at\] umich \[dot\] edu with questions.

## Trajectory Metrics

The process for computing trajectory metrics is - at a high level - as follows:

1. Run `prepare_results.py` to put files into a useful directory structure
2. Run `analyze.sh` to compute the metrics
3. Run `summarize_results.py` to extract the information from the run and print to CSV or Tex.

### Preparing Results

After running an ablation study using a combination of the `--overrides` and `--num_repeats` options, you will end up with several configurations (dicated by `--overrides`) and several trials per configuration (dictated by `--num_repeats`). All of that will be housed in a top-level experiment directory. The `prepare_results.py` script is responsible for taking information out of the top-level directory and moving it to a location that is useful for analysis. Run `python3 prepare_results.py` with the following options:

- `experiment_dir`: Path to the output stored by LONER containing all logs relavent to the experiment. LONER reports this directory when it terminates.
- `output_dir`: Where to store the processed files (for example, `./ablation_study/`)
- `dataset`: A name identifying the dataset. After this is set, output files will be placed in `<output_dir>/<dataset>`. This is useful for analyzing multiple runs, one per dataset.
- `groundtruth_traj_path`: A path to the groundtruth trajectory in TUM format. This must start with timestamps at 0.
- `--single_config` (flag, optional): Set this if there was only one configuration (i.e. `--overrides` was not set or had only one option)
- `--single_trial` (flag, optional): Set this if only one trial happened per configuration (i.e. `--num_repeats` was not used or set to 1)


A common use-case is to test the algorithm on several datasets as follows:

`python run_rosbag.py ../cfg/fusion_portable/canteen.yaml --overrides ../cfg/ablation_study.yaml --num_repeats 5`
`python run_rosbag.py ../cfg/fusion_portable/garden.yaml --overrides ../cfg/ablation_study.yaml --num_repeats 5`
`python run_rosbag.py ../cfg/fusion_portable/mcr.yaml --overrides ../cfg/ablation_study.yaml --num_repeats 5`

Then you'd prepare results like this:

`python3 prepare_results.py ../outputs/canteen_<timestamp> ./ablation_study canteen <path_to_groundtruth>`
`python3 prepare_results.py ../outputs/garden_<timestamp> ./ablation_study garden <path_to_groundtruth>`
`python3 prepare_results.py ../outputs/mcr_<timestamp> ./ablation_study mcr <path_to_groundtruth>`


### Running Analysis

Run `cd <output_dir>` then `source analyze.sh`, where `<output_dir>` was generated above.

If you used a number of trials other than 5, you'll have to change the top line of `analyze.sh` accordingly.


### Summarizing Results

Once the previous step worked, all that's left is to dump the information to a human-readable format. First, modify the following four variables in `summarize_results.py`:

1. `configs`: This should match the list of configurations within `<output_dir>/<dataset>`, and will commonly be `<config_0>,...,<config_N>`.
2. `config_names`: This is arbitrary, and only useful for human-readability. Setting `algnames=algs` is a reasonable choice if you don't want to set other names.
3. `datasets`: This is the list of datasets. It should match the folders in `<output_dir>`
4. `metrics`: Choose any combination of the metrics reported by `evo_traj`. Just `ape` is usually all you need here.

The results will be stored in csv and tex files in `<output_dir>`.


## Map Metrics

The map metrics are trickier, and the workflow is even worse. 

### Computing L1 Depth

Run `python3 analysis/compute_l1_depth.py <experiment_directory>` to do the computation. The following options are available:

- `--single_threaded`: By default, this will parallelize across all GPUs. If this flag is set, will only use one GPU.
- `--ckpt_id`: If for some reason you don't want to use the final checkpoint, you can set this.
- `--num_frames`: How many frames to compute the L1 depth for. Defaults to 25.
- `--use_est_poses`: If set, uses LONER's estimates of pose rather than groundtruth poses.

### Computing PointCloud Metrics

First, compute a mesh for all of the trials. I use this command, using `canteen` as an example (change to config for the dataset you tested on). This assumes your working directory is `LonerSLAM/analysis/compute_metrics`

```
for f in $(find <experiment_dir> -name "trial*"); do python3 ../meshing.py $f ../cfg/fusion_portable/canteen.yaml --use_weights --skip_step 1 --resolution 0.05 --save --level 0.05; done
```

Refer to `meshing.py` for details on the options.


Next, convert the meshes to pointclouds. Resolution is generally .01 or .05.

```
for f in $(find <experiment_dir> -name "*.ply"); do python3 maps/mesh_to_pcd.py $f <resolution>; done
```

Finally, compute the metrics. You need to call `evaluate_lidar_map.py` on each of the files generated from `mesh_to_pcd.py` - it's pretty easy to write bash scripts for this. The arguments to `evaluate_lidar_map.py` are as follows:

- `experiment_directory`: The experiment directory output from LONER. This needs to point to the individual trial, not the top-level directory.
- `gt_map`: A pointcloud that is groundtruth
- `--gt_trajectory`: Recommended: If set, uses this to compute an initial alignment between the groundtruth and estimated maps. TUM format.
- `--estimated_map`: A path to the pcd generated by `mesh_to_pcd.py`
- `--est_traj`: If set, uses this to help compute an initial alignment. Defaults to assuming this starts at the identity, which is generally good enough.
- `--initial_transform`: If not using `--gt_trajectory` and `--est_traj`, uses this 4x4 homogenous transform as an initial transform to apply to the estimated map. This can be found via rough alignment in CloudCompare or similar.


### Summarizing Results

From within `analysis/compute_metrics/maps`, first run `prepare_results.py`. The arguments are exactly the same as in the section on trajectory evaluation. Note that in this case, `experiment_directory` is the top-level output from LONER, not an individual trial. Next, skip straight to running `summarize_results.py`, exactly as in the trajectory evaluation section.

