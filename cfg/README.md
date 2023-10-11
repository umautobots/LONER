
## Settings

Our settings system is designed to make it easy to test with a wide variety of configurations. As a result, it's a bit complicated. Settings are processed as follows.

Each sequence has a file specifying the settings for that sequence. For example, `cfg/fusion_portable/canteen.yaml`. This specifies the following:
1. `baseline`: A path (relative to `cfg/`) with the settings for this trial to be based on, such as `fusion_portable/defaults.yaml`. 
2. Paths to the dataset, calibration, and groundtruth trajectory. The groundtruth trajectory is used for evaluation, in case you want to run with GT poses, and to pre-compute the world cube.
3. `experiment_name`: Specifies the default prefix for output directories.
4. `changes`: This is a dictionary specifying which settings to override in the defaults. See `cfg/fusion_portable/mcr.yaml` for an example of this in use.

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



Now, say we add an overrides file:

```
## File: overrides.yaml

system:
  image_resolution:
    x: [5, 10]
    y: [50, 75, 150]
```

If we specify `--overrides overrides.yaml`, the algorithm will now run 5 times. All will have `system.number_of_keyframes = 5`. The five runs will have the following combinations of x and y resolutions:

1. x = 5; y = 100
2. x = 10; y = 100
3. x = 15; y = 50
4. x = 15; y = 75
5. x = 15; y = 150

In each case, the `x` and `y` default to 15 and 100, and then one parameter is changed per run.

If you use `--overrides overrides.yaml` AND `--run_all_combos`, 6 trials will be run representing all possible combinations of x and y resolutions.

Additionally, a single overrides file can provide multiple combinations of settings, like this:

```
File: multi_overrides.yaml

-
  system:
    image_resolution:
      x: [5,10]
      y: [50,75,100]
-
  number_of_keyframes: [1,5,10]
```

In this case, the system will run in sequence on each item in the list. So, it will first process the first entry in the list without modifying `number_of_keyframes` (exactly as above). Then, it'll move on to the other entries in the list, in this case fixing everything else at default and sweeping over `number_of_keyframes`.

Finally, if you use `--num_repeats N`, every configuration will be run N times.