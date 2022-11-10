from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from common.image import Image
from common.lidar_scan import LidarScan
from common.settings import Settings
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

        print("settings: ", self._settings)
        mp.set_start_method('spawn')

        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        self._rgb_queue = mp.Queue()
        self._lidar_queue = mp.Queue()

        # The tracker inserts Frames, and the mapper reads them
        self._frame_queue = mp.Queue()

        self._mapper = Mapper(self._settings.mapper, self._frame_queue)
        self._tracker = Tracker(self._settings.tracker, self._rgb_queue, self._lidar_queue, self._frame_queue)

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
        
    # TODO: Clean up, this is horrifically sketchy
    def Stop(self):
        self._tracking_process.kill()
        self._mapping_process.kill()

    def Join(self):
        self._tracking_process.join()
        self._mapping_process.join()

    def ProcessLidar(self, lidar_scan: LidarScan) -> None:
        self._lidar_queue.put(lidar_scan)

    def ProcessRGB(self, image: Image) -> None:
        self._rgb_queue.put(image)
