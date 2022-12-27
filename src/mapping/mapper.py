import torch
import os
import torch.multiprocessing as mp

from common.pose_utils import WorldCube
from common.frame import Frame
from common.settings import Settings
from common.signals import Signal, StopSignal
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer

DEBUG_DETECT_ANOMALY = False

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

        self._world_cube = world_cube.to(settings.device, clone=True)

        self._keyframe_manager = KeyFrameManager(
            settings.keyframe_manager, settings.device)

        settings["optimizer"]["log_directory"] = settings.log_directory
        self._optimizer = Optimizer(
            settings.optimizer, calibration, self._world_cube, settings.device)

        self._term_signal = mp.Value('i', 0)
        self._processed_stop_signal = mp.Value('i', 0)

    # Spins by reading frames from the @m frame_slot as inputs.
    def run(self) -> None:

        if DEBUG_DETECT_ANOMALY:
            torch.autograd.set_detect_anomaly(True)
        
        torch.backends.cudnn.enabled = True

        self.has_written = False
        while True:
            if self._frame_slot.has_value():
                new_frame = self._frame_slot.get_value()

                if isinstance(new_frame, StopSignal):
                    break

                new_keyframe = self._keyframe_manager.process_frame(new_frame)
                
                accepted_frame = new_keyframe is not None

                accepted_str = "Accepted" if accepted_frame else "Didn't accept"
                # print(
                #     f"{accepted_str} frame at time {new_frame.start_image.timestamp}")

                if self._settings.optimizer.enabled and accepted_frame:
                    active_window = self._keyframe_manager.get_active_window()

                    self._optimizer.iterate_optimizer(active_window)

                    ckpt = {'global_step': self._optimizer._global_step,
                            'network_state_dict': self._optimizer._model.state_dict(),
                            'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                            'poses': self._keyframe_manager.get_poses_state(),
                            'occ_model_state_dict': self._optimizer._occupancy_grid_model.state_dict(),
                            'occ_optimizer_state_dict': self._optimizer._occupancy_grid_optimizer.state_dict()}

                    os.makedirs(f"{self._settings.log_directory}/checkpoints", exist_ok=True)
                    torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._global_step}.tar")
                    
        self._processed_stop_signal.value = True
        print("Mapping Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting mapping process.")
