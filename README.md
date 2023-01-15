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

Core algorithm code goes in `src`. This should be totally agnostic to any input data source.

Classes that are only used in mapping/tracking go in their respective folders. Everything else (shared classes, utils, etc) goes in `common`. 

Anything to actually run the algorithm and produce outputs goes in `examples`.

## Docs
Doxygen is set up in this repo, in case we want to use it. To build the docs:

```
cd docs
doxygen Doxyfile
```

then view the docs by opening docs/html/index.html in your favorite browser.



### Running the Code

To run the algorithm on FusionPortable, do the following

```
cd examples
python3 run_fusion_portable.py ../../data/fusion_portable/20220216_canteen_day/20220216_canteen_day_ref.bag ../../data/fusion_portable/calibration/20220209_calib/ <experiment_name>
```

This assumes your data is in ~/data/ in the container, as is done by the VSCode devcontainer. If you put it somewhere else, you'll have to change the command.

`<experiment_name>` names the output folder. All outputs are stored in `outputs/<experiment_name>_<timestamp>` where the timestamp is formatted as MMDDYY_HHMMSS.


To render outputs, use the renderer.py example. Note that this desperately needs to be updated since it's still only using camera poses used in training.