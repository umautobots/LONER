from ast import List
import time
from typing import Union

import torch
import torch.multiprocessing as mp
import os
import pickle
import yaml
import datetime
from torch.profiler import ProfilerActivity, profile


from common.pose import Pose
from common.pose_utils import compute_world_cube, WorldCube
from common.signals import Slot, Signal, StopSignal
from common.sensors import Image, LidarScan
from common.settings import Settings
from mapping.mapper import Mapper
from tracking.tracker import Tracker
from src.logging.default_logger import DefaultLogger
from pathlib import Path


class ClonerSLAM:
    """ Top-level SLAM module.

    To run sychronously, call Start with sychronous=true, then pass in measurements in the
    same or a different thread. When you're done, call Stop()
    """

    def __init__(self, settings: Union[Settings, str]) -> None:
        if isinstance(settings, str):
            self._settings = Settings.load_from_file(settings)

        elif type(settings).__name__ == "Settings":  # Avoiding strange attrdict behavior
            self._settings = settings
        else:
            raise RuntimeError(
                f"Can't load settings of type {type(settings).__name__}")

        self._single_threaded = self._settings.system.single_threaded

        if not self._single_threaded:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as e:
                # Don't blame me, pytorch should've made this something other than a generic RuntimeError
                if str(e) == "context has already been set":
                    pass
                else:
                    raise e
            
        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        self._rgb_signal = Signal(synchronous=True, single_process=self._single_threaded)
        self._lidar_signal = Signal(synchronous=True, single_process=self._single_threaded)

        # The tracker inserts Frames, and the mapper reads them
        self._frame_signal = Signal(single_process=self._single_threaded)

        # The Mapper sends updated keyframe poses each time the optimization is completed
        self._keyframe_update_signal = Signal(single_process=self._single_threaded)

        # Placeholder for the Mapping and Tracking processes 
        self._mapper = None
        self._tracker = None
        self._tracking_process = None
        self._mapping_process = None

        self._world_cube = None

        # To initialize, call initialize
        self._initialized = False        

    def initialize(self, camera_to_lidar: torch.Tensor, all_lidar_poses: torch.Tensor,
                              K_camera: torch.Tensor, camera_range: List,
                              image_size: torch.Tensor,
                              dataset_path: str,
                              ablation_name: str = None,
                              trial_idx: int = None):
        self._world_cube = compute_world_cube(
            camera_to_lidar, K_camera, image_size, all_lidar_poses, camera_range, padding=0.3)
        self._initialized = True
        self._dataset_path = Path(dataset_path).resolve().as_posix()
        
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")

        if "experiment_name" in self._settings:
            expname = self._settings.experiment_name
        else:
            expname = "experiment"
            
        self._experiment_name = f"{expname}_{now_str}"
        if ablation_name is None:
            self._log_directory = os.path.expanduser(f"~/ClonerSLAM/outputs/{self._experiment_name}/")
        else:
            self._log_directory = os.path.expanduser(f"~/ClonerSLAM/outputs/{ablation_name}/trial_{trial_idx}")


        os.makedirs(self._log_directory, exist_ok=True)

    def get_world_cube(self) -> WorldCube:
        return self._world_cube

    def start(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Can't Start: System Uninitialized. You must call initialize first.")

        # TODO: Add back support for ROS
        self._logger = DefaultLogger(self._frame_signal,
                                     self._keyframe_update_signal,
                                     self._world_cube,
                                     self._settings.calibration,
                                     self._log_directory)

        self._settings["experiment_name"] = self._experiment_name
        self._settings["dataset_path"] = self._dataset_path
        self._settings["log_directory"] = self._log_directory
        self._settings["world_cube"] = {"scale_factor": self._world_cube.scale_factor, "shift": self._world_cube.shift}

        self._settings["mapper"]["experiment_name"] = self._experiment_name
        self._settings["mapper"]["log_directory"] = self._log_directory

        self._settings["tracker"]["experiment_name"] = self._experiment_name
        self._settings["tracker"]["log_directory"] = self._log_directory

        # Pass debug settings through
        for key in self._settings.debug.flags:
            self._settings["debug"][key] = self._settings.debug.flags[key] and self._settings.debug.global_enabled

        self._settings["mapper"]["debug"] = self._settings.debug
        self._settings["tracker"]["debug"] = self._settings.debug

        world_cube = self._world_cube.as_dict()

        with open(f"{self._log_directory}/world_cube.yaml", "w+") as f:
            yaml.dump(world_cube, f)

        with open(f"{self._log_directory}/full_config.yaml", 'w+') as f:
            yaml.dump(self._settings, f)

        with open(f"{self._log_directory}/full_config.pkl", 'wb+') as f:
            pickle.dump(self._settings, f)

        if self._settings.debug.profile:
            prof_dir = f"{self._settings.log_directory}/profile"
            os.makedirs(prof_dir, exist_ok=True)
            
            self._profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{prof_dir}/tensorboard/"))
            
            self._profiler.start()

        self._mapper = Mapper(self._settings.mapper,
                              self._settings.calibration,
                              self._frame_signal,
                              self._keyframe_update_signal,
                              self._world_cube)
        self._tracker = Tracker(self._settings,
                                self._rgb_signal,
                                self._lidar_signal,
                                self._frame_signal)

        print("Starting Cloner SLAM")

        if not self._single_threaded:
            # Start the children
            self._tracking_process = mp.Process(target=self._tracker.run)
            self._mapping_process = mp.Process(target=self._mapper.run)
            self._tracking_process.daemon = True
            self._mapping_process.daemon = True
            self._tracking_process.start()
            self._mapping_process.start()

    # Stop the processes running the mapping and tracking
    def stop(self):
        
        if not self._single_threaded:
            print("Stopping ClonerSLAM Sub-Processes")
            
            self._lidar_signal.emit(StopSignal())
            self._rgb_signal.emit(StopSignal())

            while not self._tracker._processed_stop_signal.value:
                self._logger.update()
                time.sleep(0.1)
            print("Processed tracking stop")

            # Once we're done tracking frames (no new ones will be emitted),
            # we can kill the mapper.
            self._frame_signal.emit(StopSignal())
            while not self._mapper._processed_stop_signal.value:
                self._logger.update()
                time.sleep(0.1)
            print("Processed mapping stop")
        
        if self._settings.debug.profile:
            self._profiler.stop()

        self._logger.finish()

        if not self._single_threaded:
            self._tracker._term_signal.value = True
            self._mapper._term_signal.value = True
            
            self._tracking_process.join()
            self._mapping_process.join()
            print("Sub-processes Exited")



    # For use in single-threaded system. 
    def _system_update(self):
        assert self._single_threaded, "_system_update should only be called in single-threaded mode"

        self._tracker.update()
        self._mapper.update()

        if self._settings.debug.profile:
            self._profiler.step()


    def process_lidar(self, lidar_scan: LidarScan) -> None:
        assert torch.all(torch.diff(lidar_scan.timestamps) >= 0), "sort your points by timestamps!"
        self._logger.update()
        self._lidar_signal.emit(lidar_scan)
        
        if self._single_threaded:
            self._system_update()
            
    def process_rgb(self, image: Image, gt_pose: Pose=None) -> None:
        self._logger.update()
        self._rgb_signal.emit((image, gt_pose))

        if self._single_threaded:
            self._system_update()

    # Note: This is only needed when using mp.Queue instead of mp.Manager().Queue() in Slots. 
    #       Left here mainly in case that gets changed for some reason in the future and someone
    #       is confused why this exists.
    def cleanup(self):
        print("Cleaning Up Cloner SLAM")
        self._frame_signal.flush()
        self._rgb_signal.flush()
        self._lidar_signal.flush()
