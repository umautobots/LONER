# **LONER**: **L**iDAR **O**nly **Ne**ural **R**epresentations for Real-Time SLAM

### [Paper](https://arxiv.org/abs/2309.04937) | [Project Page](https://bit.ly/loner_slam)

<div align="center">
<img src="./assets/Fig1.png" width="70%" />
</div>

<p align="center">
<strong>Seth Isaacson*</strong>, <strong>Pou-Chun Kung*</strong>, <strong>Mani Ramanagopal</strong>, <strong>Ram Vasudenvan</strong>, <strong>Katherine A. Skinner</strong> <br>
{sethgi, pckung, srmani, ramv, kskin}@umich.edu
</p>

**Abstract**: *This paper proposes LONER, the first real-time LiDAR SLAM algorithm that uses a neural implicit scene representation. Existing implicit mapping methods for LiDAR show promising results in large-scale reconstruction, but either require groundtruth poses or run slower than real-time. In contrast, LONER uses LiDAR data to train an MLP to estimate a dense map in real-time, while simultaneously estimating the trajectory of the sensor. To achieve real-time performance, this paper proposes a novel information-theoretic loss function that accounts for the fact that different regions of the map may be learned to varying degrees throughout online training. The proposed method is evaluated qualitatively and quantitatively on two open-source datasets. This evaluation illustrates that the proposed loss function converges faster and leads to more accurate geometry reconstruction than other loss functions used in depth-supervised neural implicit frameworks. Finally, this paper shows that LONER estimates trajectories competitively with state-of-the-art LiDAR SLAM methods, while also producing dense maps competitive with existing real-time implicit mapping methods that use groundtruth poses.*


## Contact Information

For any questions about running the code, please open a GitHub issue and provide a detailed explanation of the problem including steps to reproduce, operating system details, and hardware. Please open issues with feature requests, we're happy to help you fit the code to your needs!

For research inquiries, please contact one of the lead authors:

- Seth Isaacson: sethgi [at] umich [dot] edu
- Pou-Chun (Frank) Kung: pckung [at] umich [dot] edu


## Running the Code

### Prerequisites
This has been tested on an Ubuntu 20.04 docker container. We highly recommend you use our docker configuration. If you have specific needs for running outside of docker, please open an issue and we'll work on documentation for how to do that. You will need:

1. Docker: https://docs.docker.com/engine/install/
2. nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Using the Docker Container
This repository has everything you need to run with docker. 


#### Building the container

```
cd <project_root>/docker
./build.sh
```

This will pull an [image](https://hub.docker.com/layers/sethgi/loner/base_1.0/images/sha256-b86796e44ccac26bcaa914a20bd16bf2068bfbb8278761b8f51d71e16fcdd6f5?context=repo) from DockerHub, then make local modifications for the user.


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
4. To make python recognize everything properly, go to the python environment extension (python logo in the left toolbar) and change the environment to Conda Base 3.8.12.

The DevContainer provided with this package assumes that datasets are stored in `~/Documents/LonerSlamData`. If you put the data somewhere else, modify the `devcontainer.json` file to point to the correct location.

When you launch the VSCode DevContainer, you might need to point VSCode manually to the workspace when prompted. It's in `/home/$USER/LonerSLAM`


## Running experiments

### Download Fusion Portable
Download the sequences you care about from [fusion portable](https://fusionportable.github.io/dataset/fusionportable), along with the 20220209 calibration.
We have tested on 20220216_canteen_day, 20220216_garden_day, and 20220219_MCR_normal_01.

Put them in the folder you pointed the docker scripts (or VSCode `devcontainer.json` file) to mount (by default `~/Documents/LonerSlamData`). Also, download the groundtruth data.

Now, modify `cfg/fusion_portable/<sequence_name>.yaml` to point to the location of the data, as seen from within the Docker container. So, if you clone the data to `~/Documents/LonerSlamData/<sequence>`, docker will mount that to `~/data/<sequence>`.

Finally, `cd examples` and `python3 run_loner.py ../../cfg/<sequence_name>.yaml`.

The results will be stored into `outputs/<experiment_name>_<timestamp>/` where `<experiment_name>` is set in the configuration file, and `<timestamp>=YYMMDD_hhmmss`

### Run

Run the canteen sequence in Fusion Portable:

```
cd examples
python3 run_loner.py ../cfg/fusion_portable/canteen.yaml
```

There are several ways to visualize the results. Each will add new files to the output folder.

#### Visualize Map
Render depth images:
```
python3 renderer.py ../outputs/<output_folder> 
```
The `renderer.py` file is also capable of producing videos by adding the `--render_video` flag. Run `renderer.py --help` for a full description of the options. 

#### Meshing the scene:
```
python3 meshing.py ../outputs/<output_folder> ../cfg/fusion_portable/canteen.yaml \
 --resolution 0.2 --skip_step 3 --level 0.1 --viz --save
```

#### Render a lidar point cloud:
This will render a LiDAR point cloud from the frame of the last. This will render LiDAR point clouds from every Nth KeyFrame, then assemble them. N defaults to 5, but can be set with --skip_step N.
```
python3 renderer_lidar.py ../outputs/<output_folder> --voxel_size 0.1
```

Results will be stored in each output folder.

#### Visualize Trajectory
##### 2D visualization using matplotlib
```
python3 plot_poses.py ../outputs/<output_folder>
```

A plot will be stored in `poses.png` in the output folder.

##### 3D visualization using evo

Download the groundtruth trajectories from the dataset in TUM format. Fusion Portable provides those [here](http://filebrowser.ram-lab.com/share/S-2Th4iV). Put them in `<sequence_folder>/ground_truth_traj.txt`:

```
mv <path_to_groundtruth>/traj/20220216_canteen_day.txt \
      ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt 
```

Then prepare the output files:
```
mkdir results && cd results
python3 ~/LonerSLAM/analysis/compute_metrics/traj/prepare_results.py \
      ~/LonerSLAM/outputs/<output_folder>\
      eval_traj canteen \
       ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt  \
       --single_trial --single_config
```

Run trajectory evaluation and visualization:
```
cd results
evo_traj tum ./eval_traj/canteen/stamped_traj_estimate0.txt \
      --ref ./eval_traj/canteen/stamped_groundtruth.txt \
      -a --t_max_diff 0.1 -p
```

This is the barebones way to produce plots, but there are lots of options for qualitative and quantitative comparisons. See the [metrics readme](analysis/compute_metrics/README.md) for more details on computing metrics.

#### Analyzing the Results
Dense trajectories are stored to `<output_dir>/trajectory`. This will contain three files:

1. `estimated_trajectory.txt`: This is the dense trajectory as estimated by LonerSLAM. 
2. `keyframe_trajectory.txt`: This includes only the keyframe poses.
3. `tracking_only.txt`: This is the result of accumulating tracking before any optimization occurs.

To compute metrics, see the information in the [metrics readme](analysis/compute_metrics/README.md) for more details on computing metrics.

## BibTeX

This work has been accepted for publication in the IEEE Robotics and Automation Letters. Please cite the pre-print version until the RA-L publication is released:

```
@misc{loner2023,
      title={LONER: LiDAR Only Neural Representations for Real-Time SLAM}, 
      author={Seth Isaacson and Pou-Chun Kung and Mani Ramanagopal and Ram Vasudevan and Katherine A. Skinner},
      year={2023},
      eprint={2309.04937},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## License


<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="http://github.com/umautobots/loner">LONER</a> by Ford Center for Autonomous Vehicles at the University of Michigan is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>

For inquiries about commercial licensing, please reach out to the authors.