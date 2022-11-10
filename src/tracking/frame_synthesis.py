from typing import Union

from common.frame import Frame
from common.image import Image
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

        self._active_frames = []

    ## Enqueues lidar data from @p lidar_scan.
    def ProcessLidar(self, lidar_scan: LidarScan) -> None:
        new_frame = Frame(None, None, lidar_scan, None)
        self._active_frames.append(new_frame)

    ## Enqueues image from @p image. 
    def ProcessImage(self, image: Image) -> None:
        print(f"Processing Image: {image}")

    def HasFrame(self) -> bool:
        return len(self._active_frames) != 0

    ## Return and remove a newly synthesized Frame. If unavailable, returns None.
    def PopFrame(self) -> Union[Frame, None]:

        # note: this is done with queues to avoid potentially expensive copies
        # which would be needed to avoid active_frame getting overwritten.
        if len(self._active_frames) == 0:
            return None
        return self._active_frames.pop(0)