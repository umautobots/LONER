import numpy as np
import torch
import torch.multiprocessing as mp
from common.frame import Frame
from common.lidar_scan import LidarScan
from common.settings import Settings
from mapping.mapper import Mapper


class ClonerSLAM:
    def __init__(self, settings: Settings) -> None:
        
        self._settings = settings

        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        self._rgb_queue = mp.Queue()
        self._lidar_queue = mp.Queue()

        # The tracker inserts Frames, and the mapper reads them
        self._frame_queue = mp.Queue()

        # Placeholder for the Mapping and Tracking processes
        self._mapper = Mapper()
        self._tracker = None

    def Run(self) -> None:
        pass

    def ProcessLidar(self, lidar_scan: LidarScan) -> None:
        self._lidar_queue.put(lidar_scan)

    def ProcessRGB(self, timestamp: float, rgb_data: torch.Tensor) -> None:
        self._rgb_queue.put((timestamp, rgb_data))
