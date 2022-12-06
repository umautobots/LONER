from typing import List

from common.frame import Frame
from common.settings import Settings
from mapping.keyframe import KeyFrame
from enum import Enum


class KeyFrameSelectionStrategy(Enum):
    TEMPORAL = 0


class WindowSelectionStrategy(Enum):
    MOST_RECENT = 0


class SampleAllocationStrategy(Enum):
    UNIFORM = 0


class KeyFrameManager:
    """ The KeyFrame Manager class creates and manages KeyFrames and passes 
    data to the optimizer.
    """

    # Constructor
    # settings: Settings object for the KeyFrame Manager
    def __init__(self, settings: Settings, device:int = -1) -> None:
        self._settings = settings
        self._keyframe_selection_strategy = KeyFrameSelectionStrategy[
            settings.keyframe_selection.strategy]
        self._window_selection_strategy = WindowSelectionStrategy[
            settings.window_selection.strategy]
        self._sample_allocation_strategy = SampleAllocationStrategy[
            settings.sample_allocation.strategy]
        
        self._device = device

        # Keep track of the start image timestamp
        self._last_accepted_frame_ts = None

        self._keyframes = []

    # Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame) -> bool:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        else:
            raise ValueError(
                f"Can't use unknown KeyFrameSelectionStrategy {self._keyframe_selection_strategy}")

        if should_use_frame:
            self._last_accepted_frame_ts = frame.start_image.timestamp
            new_keyframe = KeyFrame(frame, self._device)
            self._keyframes.append(new_keyframe)

        return should_use_frame

    def _select_frame_temporal(self, frame: Frame) -> bool:
        if self._last_accepted_frame_ts is None:
            return True
        
        dt = self._last_accepted_frame_ts - frame.start_image.timestamp
        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    # Selects which KeyFrames are to be used in the optimization, allocates
    # samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self) -> List[KeyFrame]:
        window_size = self._settings.window_selection.window_size

        if self._window_selection_strategy == WindowSelectionStrategy.MOST_RECENT:
            window = self._keyframes[-window_size:]
        else:
            raise ValueError(
                f"Can't use unknown WindowSelectionStrategy {self._window_selection_strategy}")

        return self.allocate_samples(window)

    def allocate_samples(self, keyframes: List[KeyFrame]):
        uniform_rgb_samples = self._settings.sample_allocation.num_rgb_uniform_samples
        avg_rgb_samples = self._settings.sample_allocation.avg_rgb_strategy_samples

        uniform_lidar_samples = self._settings.sample_allocation.num_lidar_uniform_samples
        avg_lidar_samples = self._settings.sample_allocation.avg_lidar_strategy_samples

        if self._sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
            for kf in keyframes:
                kf.num_uniform_rgb_samples = uniform_rgb_samples
                kf.num_strategy_rgb_samples = avg_rgb_samples
                kf.num_uniform_lidar_samples = uniform_lidar_samples
                kf.num_strategy_lidar_samples = avg_lidar_samples
        else:
            raise ValueError(
                f"Can't use unknown SampleAllocationStrategy {self._sample_allocation_strategy}")
