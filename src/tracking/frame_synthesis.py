from typing import Union

import torch
from common.frame import Frame
from common.lidar_scan import LidarScan
from common.settings import Settings
from tracking.sky_removal import SkyRemoval


class FrameSynthesis:
    """ FrameSynthesis class to process streams of data and create frames.
    """

    ## Constructor
    # @param settings: Settings for frame synthesis
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._sky_remover = SkyRemoval(settings.sky_removal)

        self._lidar_queue = [] # To be populated with LidarScans
        self._image_queue = [] # To be populated with Tensors of RGB data

    ## Enqueues lidar data from @p lidar_scan.
    def ProcessLidar(self, lidar_scan: LidarScan) -> None:
        pass

    ## Enqueues image from @p image. 
    def ProcessImage(self, image: torch.Tensor) -> None:
        pass

    ## Return a newly synthesized Frame. If unavailable, returns None.
    def GetFrame(self) -> Union[Frame, None]:
        pass

    ## A blocking version of GetFrame that blocks until a Frame can be returned.
    def WaitForNextFrame(self) -> Frame:
        pass
  
