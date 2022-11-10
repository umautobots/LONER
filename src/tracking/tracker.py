import torch.multiprocessing as mp
from common.frame import Frame
from common.settings import Settings
from tracking.frame_synthesis import FrameSynthesis


class Tracker:
    """ Tracker: Top-level Tracking module
    
    Given streams of RGB data and lidar scans, this class is responsible for creating
    Frame instances and estimating their poses.
    """

    ## Constructor
    # @param settings: Settings object for the tracker and all contained classes
    # @param rgb_queue: A mp queue owned by the top-level Cloner-SLAM class that raw images
    #       are inserted into.
    # @param lidar_queue: Same as rgb_queue, but for lidar
    # @param frame_queue: A multiprocessing queue owned by the top-level Cloner-SLAM that
    #                     the tracker puts completed frames into. 
    def __init__(self,
                 settings: Settings,
                 rgb_queue: mp.Queue,
                 lidar_queue: mp.Queue,
                 frame_queue: mp.Queue) -> None:

        self._rgb_queue = rgb_queue
        self._lidar_queue = lidar_queue
        self._frame_queue = frame_queue
        self._settings = settings

        self._frame_synthesizer = FrameSynthesis(settings.frame_synthesis)

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def Run(self) -> None:
        while True:

            # TODO: Should these be whiles? Seems like that might cause blocking.
            if not self._rgb_queue.empty():
                new_rgb = self._rgb_queue.get()

                # TODO: Decimate
                self._frame_synthesizer.ProcessImage(new_rgb)

            if not self._lidar_queue.empty():
                new_lidar = self._lidar_queue.get()
                self._frame_synthesizer.ProcessLidar(new_lidar)

            while self._frame_synthesizer.HasFrame():
                self._frame_queue.put(self._frame_synthesizer.PopFrame())

    ## TrackFrame inputs a @p frame and estimates its pose, which is stored in the Frame.
    # @returns True if tracking was successful.
    def TrackFrame(self, frame: Frame) -> bool:
        pass