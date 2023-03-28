from typing import List, Union
import torch
from enum import Enum

from common.pose import Pose
from common.frame import Frame
from common.settings import Settings
from mapping.keyframe import KeyFrame
from mapping.optimizer import Optimizer


class KeyFrameSelectionStrategy(Enum):
    TEMPORAL = 0

class WindowSelectionStrategy(Enum):
    MOST_RECENT = 0
    RANDOM = 1
    HYBRID = 2

class KeyFrameManager:
    """ The KeyFrame Manager class creates and manages KeyFrames and passes 
    data to the optimizer.
    """

    ## Constructor
    # @param settings: Settings object for the KeyFrame Manager
    def __init__(self, settings: Settings, device: int = 'cpu') -> None:
        self._settings = settings
        self._keyframe_selection_strategy = KeyFrameSelectionStrategy[
            settings.keyframe_selection.strategy]
        self._window_selection_strategy = WindowSelectionStrategy[
            settings.window_selection.strategy]

        self._device = device

        # Keep track of the start image timestamp
        self._last_accepted_frame_ts = None

        self._keyframes: List[KeyFrame] = []

        self._global_step = 0

    ## Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame) -> KeyFrame:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        else:
            raise ValueError(
                f"Can't use unknown KeyFrameSelectionStrategy {self._keyframe_selection_strategy}")

        if should_use_frame:

            self._last_accepted_frame_ts = frame.get_time()
        
            new_keyframe = KeyFrame(frame, self._device)

            # Apply the optimizated result to the new pose
            if len(self._keyframes) > 0:
                reference_kf: KeyFrame = self._keyframes[-1]

                tracked_reference_pose = reference_kf._tracked_lidar_pose.get_transformation_matrix().detach()
                tracked_current_pose = new_keyframe._tracked_lidar_pose.get_transformation_matrix().detach()
                T_track = tracked_reference_pose.inverse() @ tracked_current_pose

                optimized_tracked_pose = reference_kf.get_lidar_pose().get_transformation_matrix().detach() @ T_track
                new_keyframe._frame._lidar_pose = Pose(optimized_tracked_pose, requires_tensor=True)

            self._keyframes.append(new_keyframe)

        return new_keyframe if should_use_frame else None 

    def _select_frame_temporal(self, frame: Frame) -> bool:
        if self._last_accepted_frame_ts is None:
            return True

        dt = frame.get_time() - self._last_accepted_frame_ts

        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    ## Selects which KeyFrames are to be used in the optimization, allocates
    # samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self) -> List[KeyFrame]:
        window_size = self._settings.window_selection.window_size

        if self._window_selection_strategy == WindowSelectionStrategy.MOST_RECENT:
            window = self._keyframes[-window_size:]
        elif self._window_selection_strategy in [WindowSelectionStrategy.RANDOM, WindowSelectionStrategy.HYBRID]:
            
            if self._window_selection_strategy == WindowSelectionStrategy.RANDOM:
                num_temporal_frames = 1
            else:
                num_temporal_frames = self._settings.window_selection.hybrid_settings.num_recent_frames
                
            num_temporal_frames = min(num_temporal_frames, len(self._keyframes), window_size)
            indices = torch.randperm(len(self._keyframes) - num_temporal_frames)[:window_size-num_temporal_frames].tolist()
                        
            # Note: This isn't great design, but it's pretty important that these indices comes last. 
            # Otherwise we might not keep it in the sample allocation step
            indices += list(range(-num_temporal_frames, 0))
            window = [self._keyframes[i] for i in indices]
        else:
            raise ValueError(
                f"Can't use unknown WindowSelectionStrategy {self._window_selection_strategy}")

        return window
    
    ## @returns a list of all the KeyFrame's Pose States, represented as dicts 
    def get_poses_state(self) -> List:
        result = []
        for kf in self._keyframes:
            result.append(kf.get_pose_state())
        return result