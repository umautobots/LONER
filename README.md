# Loner SLAM

This is Loner SLAM, a neural implicit SLAM algorithm for Camera-LiDAR fusion.


## Running the Code

### Prerequisites
This has been tested on an Ubuntu 20.04 docker container. We highly recommend you use our docker configuration. You will need:

1. Docker: https://docs.docker.com/engine/install/
2. nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. Follow these instructions: https://github.com/NVIDIA/nvidia-docker/issues/595#issuecomment-519714769.

### Using the Docker Container
This repository has everything you need to run with docker. 


#### Building the container

```
cd <project_root>/docker
./build.sh
```

If you get an error about cuda not being found, you have two options:
1. Follow these instructions https://github.com/NVIDIA/nvidia-docker/issues/595#issuecomment-519714769
2. Remove the line that installs `tiny-cuda-nn` from the dockerfile, then the build will finish properly. Start the container, install `tiny-cuda-nn`, then commit the result to the tag `loner_slam`. Then re-run with `./run.sh restart`.

#### Data Preparation
By default, we assume all data has been placed in `~/Documents/LonerSlamData`. If you have data in a different, you can go into `docker/run.sh` and change `DATA_DIR` to whatever you want. If you need multiple directories mounted, you'll need to modify the run script.


#### Run Container

To run the container, `cd docker` and `./run.sh`. The `run.sh` file has the following behavior:

- If no container is currently running, a new one will be started.
- If a container is already running, you will be attached to that. Hence, running `./run.sh` from two terminals will connect you to a single docker container.
- If you run with `./run.sh restart`, the existing container (if it exists) will be killed and removed and a new one will be started.

#### VSCode
This repo contains everything you need to use the Docker extension in VSCode. To get that to run properly:
1. Install the docker extension.
2. Reload the workspace. You will likely be prompted if you want to re-open the folder in a dev-container. Say yes.
3. If not, Click the little green box in the bottom left of the screen and select "Re-open Folder in Dev Container"
4. To make python recognize everything properly, go to the python environment extension 
(python logo in the left toolbar) and change the environment to Conda Base 3.8.12.

The DevContainer provided with this package assumes that datasets are stored in `~/Documents/LonerSlamData`. If you put the data somewhere else, modify the `devcontainer.json` file to point to the correct location.

When you launch the VSCode DevContainer, you might need to point VSCode manually to the workspace when prompted. It's in `/home/$USER/LonerSLAM`


## Running experiments

### Fusion Portable
Download the sequences you care about from https://ram-lab.com/file/site/fusionportable/dataset/fusionportable/, along with the 20220209 calibration. 
We have tested on 20220216_canteen_day, 20220216_garden_day, and 20220219_MCR_normal_01.

Put them in the folder you pointed the docker scripts (or VSCode `devcontainer.json` file) to mount (by default `~/Documents/LonerSlamData`). Also, download the groundtruth data.

Now, modify `cfg/fusion_portable/<sequence_name>.yaml` to point to the location of the data, as seen from within the Docker container. So, if you clone the data to `~/Documents/LonerSlamData/<sequence>`, docker will mount that to `~/data/<sequence>`.

Finally, `cd examples/fusion_portable` and `python3 run_fusion_portable.py ../../cfg/<sequence_name>.yaml`.

The results will be stored into `outputs/<experiment_name>_<timestamp>/` where `<experiment_name>` is set in the configuration file, and `<timestamp>=YYMMDD_hhmmss`

### Analyzing the Results
Dense trajectories are stored to `<output_dir>/trajectory`. This will contain three files:

1. `estimated_trajectory.txt`: This is the dense trajectory as estimated by LonerSLAM. 
2. `keyframe_trajectory.txt`: This includes only the keyframe poses.
3. `tracking_only.txt`: This is the result of accumulating tracking before any optimization occurs.


## Replicating Our Results

First, download the relavent sequences from Fusion Portable as described above. Run:

```
cd examples/fusion_portable
python3 run_fusion_portable.py ../../cfg/fusion_portable/<canteen/garden/mcr>.py
cd ../../analysis
python3 prepare_for_traj_eval.py <output_directory> <sequence_name> # output dir is reported by the algorithm when it terminates
python3 renderer.py <output_directory> 
```

Result will be stored in each output folder. To plot trajectories, use [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation) by copying the relavent information from `<output_dir>/rpg_evaluation` into wherever you cloned the `rpg_trajectory_evaluation` repo. In particular, move `rpg_evaluation/<experiment_name>` to `rpg_trajectory_evaluation/results` and `rpg_evaluation/<config_name>.yaml` to `rpg_trajectory_evaluation/analyze_trajectories_config` Then, run with:

```
python3 scripts/analyze_trajectories.py <config_name>.yaml --output_dir ./results/<experiment_name>/ --results_dir ./results/<experiment_name>/ --platform desktop --odometry_error_per_dataset --plot_trajectories --png
```

Note that you may need to modify `rpg_trajectory_evaluation` to allow a larger time difference when associating poses between the ground truth and the estimated trajectories. 0.15s worked for us. 



## Settings

Our settings system is designed to make it easy to test with a wide variety of configurations. As a result, it's a bit complicated. Settings are processed as follows.

Each sequence has a file specifying the settings for that sequence. For example, `cfg/fusion_portable/canteen.yaml`. This specifies the following:
1. `baseline`: A path (relative to `cfg/`) with the settings for this trial to be based on, such as `fusion_portable/defaults.yaml`. 
2. Paths to the dataset, calibration, and groundtruth trajectory. The groundtruth trajectory is used for evaluation, in case you want to run with GT poses, and to pre-compute the world cube.
3. `experiment_name`: Specifies the default prefix for output directories.
4. `changes`: This is a dictionary specifying which settings to override in the defaults. See `cfg/fusion_portable/mcr.yaml` for an example of this in use.

Next, if you specified `--lite` when running `run_fusion_portable.py`, the settings as determined above will be augmented with the settings in `cfg/Loner_slam_lite.yaml`.

Finally, if you want to run an ablation study or try sets of parameters, you may pass `--overrides <path_to_yaml>` to `run_fusion_portable.py`. See `cfg/traj_ablation.py` for an example. You specify a path (in yaml) to the parameter, then one or more possible values. By default, for each value you specified in the overrides file, one trial of the algorithm will be run. Each trial will run with the settings as determined above, but with that one additional parameter changed. You may also pass `--run_all_combos` which will try all combinations of parameters specified in the overrides.

For example, consider a simple example where the baseline settings are:

```
## File: simple_baseline.yaml

system:
  number_of_keyframes: 10
  number_of_samples: 20
  image_resolution:
    x: 10
    y: 100
```

Then, in the sequence settings you provide the following:
```
## File: sequence.yaml

baseline: simple_baseline.yaml
dataset: <path_to_data>
calibration: <path_to_calibration>
groundtruth_traj: <path_to_gt_poses>
experiment_name: simple
changes:
  system:
    image_resolution:
      x: 15
```


If you run without specifying any options, the algorithm will run once, with all default settings except for `system.image_resolution.x = 15`.

Say our `Loner_slam_lite.yaml` file then looks like this:

```
## File: Loner_slam_lite.yaml

system:
  number_of_keyframes: 5
```

Now, if we run with `--lite`, the algorithm will run once with default settings, but `system.image_resolution.x = 15` and `system.number_of_keyframes = 5`.


Now, say we add an overrides file:

```
## File: overrides.yaml

system:
  image_resolution:
    x: [5, 10]
    y: [50, 75, 150]
```

If we specify `--lite` AND `--overrides overrides.yaml`, the algorithm will now run 5 times. All will have `system.number_of_keyframes = 5`. The five runs will have the following combinations of x and y resolutions:

1. x = 5; y = 100
2. x = 10; y = 100
3. x = 15; y = 50
4. x = 15; y = 75
5. x = 15; y = 150

In each case, the `x` and `y` default to 15 and 100, and then one parameter is changed per run.

Finally, if you specify `--lite` AND `--overrides overrides.yaml` AND `--run_all_combos`, 6 trials will be run representing all possible combinations of x and y resolutions.


The last thing (for real this time) is you can also provide `--num_repeats=<n>`, which will run all configurations (as determined above) `n` times.