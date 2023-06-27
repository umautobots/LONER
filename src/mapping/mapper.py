import cProfile
from typing import Union
import torch
import os
import torch.multiprocessing as mp

from common.pose_utils import WorldCube
from common.frame import Frame
from common.settings import Settings
from common.signals import Signal, StopSignal
from mapping.keyframe_manager import KeyFrameManager
from mapping.optimizer import Optimizer

import time

class Mapper:
    """ Mapper is the top-level Mapping module which manages and optimizes the 
    Loner Map.

    It reads in data from the frame_slot, and uses that to build and update
    the optimizer.
    """

    ## Constructor
    # @param settings: The settings for the mapping and all contained classes
    # @param frame_signal: A Signal which the tracker emits to with completed Frame objects
    def __init__(self, settings: Settings, calibration: Settings, frame_signal: Signal,
                 keyframe_update_signal: Signal, world_cube: WorldCube,
                 enable_sky_segmentation: bool = True) -> None:
                 
        self._frame_slot = frame_signal.register()
        self._keyframe_update_signal = keyframe_update_signal

        self._settings = settings

        self._lidar_only = settings.lidar_only

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
            self._lidar_only,
            enable_sky_segmentation)

        self._term_signal = mp.Value('i', 0)
        self._processed_stop_signal = mp.Value('i', 0)
        os.makedirs(f"{self._settings.log_directory}/checkpoints", exist_ok=True)
        self.last_ckpt = {}

        self._last_mapped_frame_time = None

    ## A single iteration of mapper: Checks if a new frame is available. If so,
    # decides if it's a keyframe, then if so performs optimization
    def update(self) -> None:
        tic = time.time()
        if self._processed_stop_signal.value:
            print("Not updating mapper: Mapping already done.")
        
        did_map_frame = False

        if self._frame_slot.has_value():
            new_frame: Union[StopSignal, Frame] = self._frame_slot.get_value()

            if isinstance(new_frame, StopSignal):
                self._processed_stop_signal.value = 1
                return
            
            if self._settings.debug.use_groundtruth_poses:
                new_frame._lidar_pose = new_frame._gt_lidar_pose

            new_keyframe = self._keyframe_manager.process_frame(new_frame)
            
            accepted_frame = new_keyframe is not None
 
            # print(f"{accepted_str} frame at time {image_ts}")
        
            if self._settings.optimizer.enabled and accepted_frame:                                
                active_window = self._keyframe_manager.get_active_window()

                if self._last_mapped_frame_time is not None:    
                    self._last_mapped_frame_time.value = new_keyframe.get_time()

                self._optimizer.iterate_optimizer(active_window)

                pose_state = self._keyframe_manager.get_poses_state()

                kf_idx = self._optimizer._keyframe_count - 1

                if isinstance(self._settings.log_level, tuple):
                    self._settings.log_level = self._settings.log_level[0]
            
                if (kf_idx % 10 == 0 and self._settings.log_level == "STANDARD") or self._settings.log_level == "VERBOSE":
                    if self._settings.optimizer.samples_selection.strategy == 'OGM':
                        ckpt = {'global_step': self._optimizer._global_step,
                                'network_state_dict': self._optimizer._model.state_dict(),
                                'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                                'poses': pose_state,
                                'occ_model_state_dict': self._optimizer._occupancy_grid_model.state_dict(),
                                'occ_optimizer_state_dict': self._optimizer._occupancy_grid_optimizer.state_dict()}
                    else:
                        ckpt = {'global_step': self._optimizer._global_step,
                                'network_state_dict': self._optimizer._model.state_dict(),
                                'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                                'poses': pose_state}

                    torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{kf_idx}.tar")

                else:
                    ckpt = {'global_step': self._optimizer._global_step,
                            'poses': pose_state}
                    torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{kf_idx}.tar")
                
                self._keyframe_update_signal.emit(pose_state)
                did_map_frame = True
                
                
            elif not self._settings.optimizer.enabled:
                if self._optimizer._global_step % 100 == 0:
                    pose_state = self._keyframe_manager.get_poses_state()
                    ckpt = {'poses': pose_state}
                    torch.save(ckpt, f"{self._settings.log_directory}/checkpoints/ckpt_{self._optimizer._keyframe_count}.tar")
                self._optimizer._global_step += 1

        toc = time.time()

        if did_map_frame and self._settings.debug.log_times:
            with open(f"{self._settings.log_directory}/map_times.csv", "a+") as time_f:
                time_f.write(f"{toc - tic}\n")

    ## Spins by reading frames from the @m frame_slot as inputs.
    def run(self, last_mapped_frame_time) -> None:
        self._last_mapped_frame_time = last_mapped_frame_time

        if self._settings.debug.pytorch_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        self.has_written = False
        while not self._processed_stop_signal.value:
            self.update()
        
        self.finish()
        
        print("Mapping Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting mapping process.")

    def finish(self):

        # Hack: Train RGB MLP
        # print("Training RGB MLP at shutdown")
        # optimizer_settings = OptimizationSettings(3, 10_000, True, False, True, False)
        # kf_window = self._keyframe_manager._keyframes
        # self._optimizer.iterate_optimizer(kf_window, optimizer_settings)

        pose_state = self._keyframe_manager.get_poses_state()

        if self._settings.optimizer.samples_selection.strategy == 'OGM':
            last_ckpt = {'global_step': self._optimizer._global_step,
                    'network_state_dict': self._optimizer._model.state_dict(),
                    'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                    'poses': pose_state,
                    'occ_model_state_dict': self._optimizer._occupancy_grid_model.state_dict(),
                    'occ_optimizer_state_dict': self._optimizer._occupancy_grid_optimizer.state_dict()}
        else:
            last_ckpt = {'global_step': self._optimizer._global_step,
                    'network_state_dict': self._optimizer._model.state_dict(),
                    'optimizer_state_dict': self._optimizer._optimizer.state_dict(),
                    'poses': pose_state}

        print("Saving Last Checkpoint to", f"{self._settings.log_directory}/checkpoints/final.tar")
        torch.save(last_ckpt, f"{self._settings.log_directory}/checkpoints/final.tar")
