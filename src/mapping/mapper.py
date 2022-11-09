import torch.multiprocessing as mp

from common.frame import Frame
from common.settings import Settings
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer


class Mapper:
    """ Mapper is the top-level Mapping module which manages and optimizes the 
    CLONeR Map.

    It reads in data from the frame_queue, and uses that to build and update
    the optimizer.
    """

    ## Constructor
    # @param settings: The settings for the mapping and all contained classes
    # @param frame_queue: A multiprocessing queue of frames owned by the top-level
    #      ClonerSLAM module which is written to by the Tracker and read by Mapper.
    def __init__(self, settings: Settings, frame_queue: mp.Queue) -> None:
        self._frame_queue = frame_queue
        self._settings = settings

        self._keyframe_manager = KeyFrameManager(settings.keyframe_manager)
        self._optimizer = Optimizer(settings.optimizer)

    ## Spin by reading frames from the @m frame_queue as inputs.
    def Run(self) -> None:
        pass