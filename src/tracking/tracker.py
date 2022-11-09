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
    # @param frame_queue: A multiprocessing queue owned by the top-level Cloner-SLAM that
    #                     the tracker puts completed frames into. 
    def __init__(self, settings: Settings, frame_queue: mp.Queue) -> None:
        self._frame_queue = frame_queue
        self._settings = settings

        self._frame_synthesizer = FrameSynthesis(settings.frame_synthesis)

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def Run(self) -> None:
        pass

    ## TrackFrame inputs a @p frame and estimates its pose, which is stored in the Frame.
    # @returns True if tracking was successful.
    def TrackFrame(self, frame: Frame) -> bool:
        pass