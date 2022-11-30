from typing import Dict

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
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._keyframe_selection_strategy = KeyFrameSelectionStrategy(settings.key_frame_selection.strategy)
        self._window_selection_strategy = WindowSelectionStrategy(settings.window_selection.strategy)
        self._sample_allocation_strategy = SampleAllocationStrategy(settings.sample_allocation_strategy)

        # Keep track of the start image timestamp
        self._last_accepted_frame_ts = None

        self._keyframes = []

    # Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame) -> bool:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        else:
            raise ValueError(f"Can't use unknown KeyFrameSelectionStrategy {self._keyframe_selection_strategy}")

        if should_use_frame:
            self._last_accepted_frame_ts = frame.start_image.timestamp
            new_keyframe = KeyFrame(frame)
            self._keyframes.append(new_keyframe)

    def _select_frame_temporal(self, frame: Frame) -> bool:
        dt = self._last_accepted_frame_ts - frame.start_image.timestamp
        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    # Selects which KeyFrames are to be used in the optimization, allocates
    # samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self) -> Dict[KeyFrame, int]:
        window_size = self._settings.window_selection.window_size

        if self._window_selection_strategy == WindowSelectionStrategy.MOST_RECENT:
            window = self._keyframes[-window_size:]
        else:
            raise ValueError(f"Can't use unknown WindowSelectionStrategy {self._window_selection_strategy}")

        return self.allocate_samples(window)

    
    def allocate_samples(self, keyframes) -> Dict[KeyFrame, int]:
        avg_num_samples = self._settings.sample_allocation.avg_num_samples_from_strategy

        if self._sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
            return {kf: avg_num_samples for kf in keyframes}
        else:
            raise ValueError(f"Can't use unknown SampleAllocationStrategy {self._sample_allocation_strategy}")