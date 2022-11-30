import torch.multiprocessing as mp

from common.frame import Frame
from common.settings import Settings
from common.signals import Signal, Slot
from common.utils import StopSignal
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer


class Mapper:
    """ Mapper is the top-level Mapping module which manages and optimizes the 
    CLONeR Map.

    It reads in data from the frame_queue, and uses that to build and update
    the optimizer.
    """

    # Constructor
    # @param settings: The settings for the mapping and all contained classes
    # @param frame_signal: A Signal which the tracker emits to with completed Frame objects
    def __init__(self, settings: Settings, frame_signal: Signal) -> None:
        self._frame_slot = frame_signal.register()
        self._settings = settings

        self._keyframe_manager = KeyFrameManager(settings.keyframe_manager)
        self._optimizer = Optimizer(settings.optimizer)

        self._term_signal = mp.Value('i', 0)
        self._processed_stop_signal = mp.Value('i', 0)

    # Spin by reading frames from the @m frame_queue as inputs.
    def run(self) -> None:
        while True:
            if self._frame_slot.has_value():
                new_frame = self._frame_slot.get_value()

                if isinstance(new_frame, StopSignal):
                    break

        self._processed_stop_signal.value = True
        print("Mapping Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting mapping process.")
