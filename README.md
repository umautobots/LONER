# Cloner-SLAM

## VSCode
This repo contains everything you need to use the Docker extension in VSCode. To get that to run properly:
1. Install the docker extension.
2. Reload the workspace. You will likely be prompted if you want to re-open the folder in a dev-container. Say yes.
3. If not, Click the little green box in the bottom left of the screen and select "Re-open Folder in Dev Container"
4. To make python recognize everything properly, go to the python environment extension 
(python logo in the left toolbar) and change the environment to Conda Base 3.8.12.

## Docker
### Build

- `cd docker`
- `./build.sh`

If you get an error about cuda not being found, you have two options:
1. Follow these instructions https://github.com/NVIDIA/nvidia-docker/issues/595#issuecomment-519714769
2. Remove the line that installs `tiny-cuda-nn` from the dockerfile, then the build will finish properly. Start the container, install `tiny-cuda-nn`, then commit the result to the tag `cloner_slam`. Then re-run with `./run.sh restart`.

## Setting up your VSCode DevContainer

The DevContainer provided with this package assumes that datasets are stored in `~/Documents/ClonerSlamData`. If you put the data somewhere else, modify the `devcontainer.json` file to point to the correct location.

When you launch the VSCode DevContainer, you might need to point VSCode manually to the workspace when prompted. It's in `/home/$USER/ClonerSLAM`

### Run Container

- `cd docker`
- `./run.sh`

## Architecture

### Code Organization

The core algorithm code goes in `src`. This should be totally agnostic to any input data source. Within `src`, classes that are only used in mapping/tracking go in their respective folders. Everything else (shared classes, utils, etc) goes in `common`. 

Anything to actually run the algorithm and produce outputs goes in `examples`. 

Scripts for analyzing performance are in `analysis`. 

Outputs are stored in `outputs`.

Unit tests are limited, but contained in `test`.

### Inter-Process Communication

The current implementation has three processes running: A main process which handles data I/O, a tracker, and a mapper. A simple Signals/Slots implementation is used to share data, which is based on Multiprocessing Queues. This implementation can be found in `src/common/signals.py`.

Note that this currently causes a decent amount of overhead, and I'm looking for an alternative.


### Settings

Each project needs three configuration files: A top-level settings, a model configuration, and a NeRF configuration. For example, see `cfg/default_settings.yaml`, `cfg/model_config/decoupled_carla_halfres.yaml`, and `cfg/nerf_config/carla_decoupled_hash.yaml`.

Settings in the top-level configuration are passed through to their respective classes and subclasses. As an example, say you want to add a new setting to the optimizer to control a coefficient in the loss. In `default_settings.yaml` you could add an entry under `mapper/optimizer` as `k_test: 0`. Then, within the optimizer, you'll immediately be able to access `self._settings.k_test`.

#### Debugging

An exception to the organization is the debug settings. All modules get access to all debug settings. You can access each flag by it's name directly, ignoring the `flags` and `global_enabled` fields. For example, say you want to add a new debug tool which prints "here!" at a certain point in the code. In the settings, under `debug/flags`, add `print_here: True`. Then, within the code (for example in `optimizer.py`), you'll directly be able to access `self._settings.debug.print_here`. The `global_enable` flag will toggle all the debug flags: `self._settings.debug.print_here` will return true if `debug/flags/print_here` AND `debug/global_enabled` are `True` in the settings.

## Docs
Doxygen is set up in this repo, in case we want to use it. To build the docs:

```
cd docs
doxygen Doxyfile
```

then view the docs by opening docs/html/index.html in your favorite browser.


## Running the Code

To run the algorithm on FusionPortable, do the following

```
cd examples
python3 run_fusion_portable.py ../../data/fusion_portable/20220216_canteen_day/20220216_canteen_day_ref.bag ../../data/fusion_portable/calibration/20220209_calib/ <experiment_name>
```

This assumes your data is in ~/data/ in the container, as is done by the VSCode devcontainer. If you put it somewhere else, you'll have to change the command.

`<experiment_name>` names the output folder. All outputs are stored in `outputs/<experiment_name>_<timestamp>` where the timestamp is formatted as MMDDYY_HHMMSS.

To render outputs, use the renderer.py example. Note that this desperately needs to be updated since it's still only using camera poses used in training.