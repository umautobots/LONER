from ast import List
from time import sleep
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from common.pose import Pose
from common.pose_utils import compute_world_cube, WorldCube
from common.signals import Slot, Signal
from common.sensors import Image, LidarScan
from common.settings import Settings
from common.utils import StopSignal
from mapping.mapper import Mapper
from tracking.tracker import Tracker

class ClonerSLAM:
    """ Top-level SLAM module.

    Options for running synchronously (all in one thread) or asynchronously.

    To run asynchronously, call Run with synchronous=false, then pass in measurements
    from another thread (i.e. using ROS callbacks)

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

        mp.set_start_method('spawn')

        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        self._rgb_signal = Signal()
        self._lidar_signal = Signal()

        # The tracker inserts Frames, and the mapper reads them
        self._frame_signal = Signal()

        # Placeholder for the Mapping and Tracking processes
        self._mapper = None
        self._tracker = None
        self._tracking_process = None
        self._mapping_process = None

        self._world_cube = None

        # To initialize, call precompute_world_cube
        self._initialized = False

    def precompute_world_cube(self, all_cam_poses: torch.Tensor, all_lidar_poses: torch.Tensor,
                              K_camera: torch.Tensor, camera_range: List,
                              image_size: torch.Tensor):
        self._world_cube = compute_world_cube(
            all_cam_poses, K_camera, image_size, all_lidar_poses, camera_range)
        self._initialized = True

    def get_world_cube(self) -> WorldCube:
        return self._world_cube

    def start(self, synchronous: bool = True) -> None:

        if not self._initialized:
            raise RuntimeError(
                "Can't Start: System Uninitialized. You must call precompute_world_cube first.")

        self._mapper = Mapper(self._settings.mapper,
                              self._settings.calibration,
                              self._frame_signal,
                              self._world_cube)
        self._tracker = Tracker(self._settings,
                                self._rgb_signal,
                                self._lidar_signal,
                                self._frame_signal,
                                self._world_cube)

        print("Starting Cloner SLAM")
        # Start the children
        self._tracking_process = mp.Process(target=self._tracker.run)
        self._mapping_process = mp.Process(target=self._mapper.run)
        self._tracking_process.daemon = True
        self._mapping_process.daemon = True
        self._tracking_process.start()
        self._mapping_process.start()

        if not synchronous:
            self._tracking_process.join()
            self._mapping_process.join()

    # Stop the processes running the mapping and tracking
    def stop(self, waiting_action=None, finish_action=None):
        print("Stopping ClonerSLAM Sub-Processes")

        self._lidar_signal.emit(StopSignal())
        self._rgb_signal.emit(StopSignal())

        while not self._tracker._processed_stop_signal.value:
            if waiting_action is not None:
                waiting_action()
            sleep(0.1)
        print("Processed tracking stop")

        # Once we're done tracking frames (no new ones will be emitted),
        # we can kill the mapper.
        self._frame_signal.emit(StopSignal())
        while not self._mapper._processed_stop_signal.value:
            if waiting_action is not None:
                waiting_action()
            sleep(0.1)
        print("Processed mapping stop")

        if finish_action is not None:
            finish_action()

        self._tracker._term_signal.value = True
        self._mapper._term_signal.value = True

        self._tracking_process.join()
        self._mapping_process.join()

        print("Sub-processes Exited")

    def process_lidar(self, lidar_scan: LidarScan) -> None:
        lidar_scan.transform_world_cube(self._world_cube)
        self._lidar_signal.emit(lidar_scan)

    def process_rgb(self, image: Image, gt_pose: Pose = None) -> None:
        self._rgb_signal.emit((image, gt_pose))

    def cleanup(self):
        print("Cleaning Up ClonerSlam")
        self._frame_signal.flush()
        self._rgb_signal.flush()
        self._lidar_signal.flush()
