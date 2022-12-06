import torch.multiprocessing as mp

from common.pose_utils import WorldCube
from common.frame import Frame
from common.settings import Settings
from common.signals import Signal
from common.utils import StopSignal
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer


class Mapper:
    """ Mapper is the top-level Mapping module which manages and optimizes the 
    CLONeR Map.

    It reads in data from the frame_slot, and uses that to build and update
    the optimizer.
    """

    # Constructor
    # @param settings: The settings for the mapping and all contained classes
    # @param frame_signal: A Signal which the tracker emits to with completed Frame objects
    def __init__(self, settings: Settings, calibration: Settings, frame_signal: Signal,
                 world_cube: WorldCube) -> None:
        self._frame_slot = frame_signal.register()
        self._settings = settings

        self._keyframe_manager = KeyFrameManager(settings.keyframe_manager, settings.device)
        self._optimizer = Optimizer(settings.optimizer, calibration, world_cube, settings.device)

        self._term_signal = mp.Value('i', 0)
        self._processed_stop_signal = mp.Value('i', 0)

        self._world_cube = world_cube

    # Spin by reading frames from the @m frame_slot as inputs.
    def run(self) -> None:
        while True:
            if self._frame_slot.has_value():
                new_frame = self._frame_slot.get_value()

                if isinstance(new_frame, StopSignal):
                    break

                accepted_frame = self._keyframe_manager.process_frame(
                    new_frame)

                # if accepted_frame:
                #     active_window = self._keyframe_manager.get_active_window()
                #     self._optimizer.iterate_optimizer(active_window)

        self._processed_stop_signal.value = True
        print("Mapping Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting mapping process.")
