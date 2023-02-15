import torch
import os
import torch.multiprocessing as mp

from common.pose_utils import WorldCube
from common.frame import Frame
from common.settings import Settings
from common.signals import Signal, StopSignal
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer


class Mapper:
    """ Mapper is the top-level Mapping module which manages and optimizes the 
    CLONeR Map.

    It reads in data from the frame_slot, and uses that to build and update
    the optimizer.
    """

    ## Constructor
    # @param settings: The settings for the mapping and all contained classes
    # @param frame_signal: A Signal which the tracker emits to with completed Frame objects
    def __init__(self, settings: Settings, calibration: Settings, frame_signal: Signal,
                 keyframe_update_signal: Signal, world_cube: WorldCube) -> None:
        self._frame_slot = frame_signal.register()
        self._keyframe_update_signal = keyframe_update_signal

        self._settings = settings

        self._world_cube = world_cube #.to('device', clone=True)

        settings["keyframe_manager"]["debug"] = settings.debug
        settings["keyframe_manager"]["log_directory"] = settings.log_directory

        self._keyframe_manager = KeyFrameManager(
            settings.keyframe_manager, 'cpu' if settings.data_prep_on_cpu else self._settings.device)

        settings["optimizer"]["debug"] = settings.debug
        settings["optimizer"]["log_directory"] = settings.log_directory
        self._optimizer = Optimizer(
            settings.optimizer, calibration, self._world_cube, 0,
            settings.debug.use_groundtruth_poses,
            settings.keyframe_manager.sample_allocation.strategy)

        self._term_signal = mp.Value('i', 0)
        self._processed_stop_signal = mp.Value('i', 0)
        os.makedirs(f"{self._settings.log_directory}/checkpoints", exist_ok=True)

    def update(self) -> None:
        if self._processed_stop_signal.value:
            print("Not updating mapper: Mapping already done.")
            
        if self._frame_slot.has_value():
            new_frame = self._frame_slot.get_value()

            if isinstance(new_frame, StopSignal):
                self._processed_stop_signal.value = 1
                return

            new_keyframe = self._keyframe_manager.process_frame(new_frame)
            
            accepted_frame = new_keyframe is not None

            accepted_str = "Accepted" if accepted_frame else "Didn't accept"
            # print(
            #     f"{accepted_str} frame at time {new_frame.start_image.timestamp}")
        
            if self._settings.optimizer.enabled and accepted_frame:

                active_window = self._keyframe_manager.get_active_window(self._optimizer)

                self._optimizer.iterate_optimizer(active_window)

                pose_state = self._keyframe_manager.get_poses_state()
                
                if self._optimizer._keyframe_count % 10 == 0 or self._settings.log_verbose:
                    ckpt = {'global_step': self._optimizer._global_step,
                            'network_state_dict': self._optimizer._model.state_dict(),
                            'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                            'poses': pose_state,
                            'occ_model_state_dict': self._optimizer._occupancy_grid_model.state_dict(),
                            'occ_optimizer_state_dict': self._optimizer._occupancy_grid_optimizer.state_dict()}
                else:
                    ckpt = {'global_step': self._optimizer._global_step,
                            # 'network_state_dict': self._optimizer._model.state_dict(),
                            # 'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                            'poses': pose_state,
                            'occ_model_state_dict': self._optimizer._occupancy_grid_model.state_dict(),
                            'occ_optimizer_state_dict': self._optimizer._occupancy_grid_optimizer.state_dict()}

                print("Sending KF Update")
                self._keyframe_update_signal.emit(pose_state)
                
                print("Saving Checkpoint to", f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._keyframe_count}.tar")
                torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._keyframe_count}.tar")
            elif not self._settings.optimizer.enabled:
                if self._optimizer._global_step % 100 == 0:
                    pose_state = self._keyframe_manager.get_poses_state()
                    ckpt = {'poses': pose_state}
                    print("Saving Checkpoint to", f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._keyframe_count}.tar")
                    torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._keyframe_count}.tar")
                self._optimizer._global_step += 1

    ## Spins by reading frames from the @m frame_slot as inputs.
    def run(self) -> None:

        if self._settings.debug.pytorch_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        self.has_written = False
        while not self._processed_stop_signal.value:
            self.update()
            
        print("Mapping Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting mapping process.")
