from time import sleep
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

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
            with open(settings, 'r') as settings_file:
                self._settings = Settings(yaml.load(settings_file, Loader=yaml.FullLoader))
        elif isinstance(settings, Settings):
            self._settings = settings
        else:
            raise RuntimeError(f"Can't load settings of type {type(settings).__name__}")

        print("Project Settings:", self._settings)
        mp.set_start_method('spawn')

        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        self._rgb_signal = Signal()
        self._lidar_signal = Signal()

        # The tracker inserts Frames, and the mapper reads them
        self._frame_signal = Signal()

        self._mapper = Mapper(self._settings.mapper, self._frame_signal)
        self._tracker = Tracker(self._settings,
                                self._rgb_signal,
                                self._lidar_signal,
                                self._frame_signal)

        # Placeholder for the Mapping and Tracking processes
        self._tracking_process = None
        self._mapping_process = None


    def Start(self, synchronous: bool = True) -> None:

        # Start the children
        self._tracking_process = mp.Process(target=self._tracker.Run)
        self._mapping_process = mp.Process(target=self._mapper.Run)
        self._tracking_process.start()
        self._mapping_process.start()

        if not synchronous:
            self._tracking_process.join()
            self._mapping_process.join()
        
    ## Stop the processes running the mapping and tracking
    def Stop(self):
        print("Stopping ClonerSLAM Sub-Processes")

        self._lidar_signal.Emit(StopSignal())
        self._rgb_signal.Emit(StopSignal())

        while not self._tracker._processed_stop_signal.value:
            sleep(0.1)
        print("Processed tracking stop")
        
        self._frame_signal.Emit(StopSignal())
        while not self._mapper._processed_stop_signal.value:
            sleep(0.1)
        print("Processed mapping stop")

        self._tracker._term_signal.value = True
        self._mapper._term_signal.value = True

        self._tracking_process.join()
        self._mapping_process.join()

    def ProcessLidar(self, lidar_scan: LidarScan) -> None:
        self._lidar_signal.Emit(lidar_scan)

    def ProcessRGB(self, image: Image) -> None:
        self._rgb_signal.Emit(image)
