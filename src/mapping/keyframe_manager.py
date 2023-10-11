"""
File: src/mapping/keyframe_manager.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

from typing import List
import torch
from enum import Enum

from common.pose import Pose
from common.frame import Frame
from common.settings import Settings
from mapping.keyframe import KeyFrame


class KeyFrameSelectionStrategy(Enum):
    TEMPORAL = 0
    MOTION=1
    HYBRID=2
    HYBRID_LAZY=3

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

        # In hybrid lazy mode, we keep track of another value.
        # If enough time has passed, but the amount of motion caused a KF to be rejected,
        # we update this timestamp. Useful for bookkeeping to keep tracking up to date
        self._last_motion_rejected_frame_ts = None

        self._keyframes: List[KeyFrame] = []

        self._global_step = 0

    def __len__(self):
        return len(self._keyframes)

    ## Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame) -> KeyFrame:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        elif self._keyframe_selection_strategy in [KeyFrameSelectionStrategy.MOTION, KeyFrameSelectionStrategy.HYBRID, 
                                                                                KeyFrameSelectionStrategy.HYBRID_LAZY]:
            motion_criteria_met = self._select_frame_motion(frame)
            temporal_criteria_met = self._select_frame_temporal(frame)

            if temporal_criteria_met and not motion_criteria_met:
                self._last_motion_rejected_frame_ts = frame.get_time()

            if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.MOTION:
                should_use_frame = motion_criteria_met
            else:
                should_use_frame = motion_criteria_met and temporal_criteria_met
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


        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.HYBRID:
            
            # In hybrid, we might need to update temporal criteria even if we don't accept the frame
            # since above if block might not be entered. 
            if temporal_criteria_met:
                self._last_accepted_frame_ts = frame.get_time()

            # This is a bit confusing. If temporal_criteria_met is True, we are going to process a KeyFrame.
            # If the motion criteria is also met, we create and return a new KeyFrame, which this logic covers.
            # If the motion criteria is NOT met, we process the past KeyFrame again.
            # So this one line covers all three cases: temporal only, temporal and motion,
            # or not temporal (so we don't care about motion)
            return self._keyframes[-1] if temporal_criteria_met else None
            
        
        return new_keyframe if should_use_frame else None 

    def get_last_mapped_time(self):
        if self._keyframe_selection_strategy in [KeyFrameSelectionStrategy.HYBRID_LAZY,
                                                 KeyFrameSelectionStrategy.MOTION] and \
                self._last_motion_rejected_frame_ts is not None:
            return max(self._last_motion_rejected_frame_ts, self._last_accepted_frame_ts)
        return self._last_accepted_frame_ts

    def _select_frame_temporal(self, frame: Frame) -> bool:
        if len(self._keyframes) == 0:
            return True

        dt = frame.get_time() - self._last_accepted_frame_ts

        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    def _select_frame_motion(self, frame: Frame) -> bool:
        if len(self._keyframes) == 0:
            return True
        
        reference_kf = self._keyframes[-1]
        reference_pose = reference_kf.get_lidar_pose()

        current_pose = frame.get_lidar_pose()

        reference_to_current = reference_pose.inv() * current_pose

        dT = reference_to_current.get_translation().norm()
        dR = reference_to_current.get_axis_angle().rad2deg().norm()

        dT_threshold = self._settings.keyframe_selection.motion.translation_threshold_m
        dR_threshold = self._settings.keyframe_selection.motion.rotation_threshold_deg

        return dT >= dT_threshold or dR >= dR_threshold
    
    def get_keyframes(self, idxs = None) -> List[KeyFrame]:
        if idxs is None:
            return self._keyframes
        return [self._keyframes[i] for i in idxs]

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