from typing import List, Union
import torch
from enum import Enum

from common.pose import Pose
from common.frame import Frame, SimpleFrame
from common.settings import Settings
from mapping.keyframe import KeyFrame
from mapping.optimizer import Optimizer


class KeyFrameSelectionStrategy(Enum):
    TEMPORAL = 0

class WindowSelectionStrategy(Enum):
    MOST_RECENT = 0
    RANDOM = 1
    HYBRID = 2

class SampleAllocationStrategy(Enum):
    UNIFORM = 0
    ACTIVE = 1
    ACTIVE_LIDAR = 2
    ACTIVE_CAMERA = 3

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
        self._rgb_sample_allocation_strategy = SampleAllocationStrategy[
            settings.sample_allocation.rgb_strategy]
        self._lidar_sample_allocation_strategy = SampleAllocationStrategy[
            settings.sample_allocation.lidar_strategy]

        self._device = device

        # Keep track of the start image timestamp
        self._last_accepted_frame_ts = None

        self._keyframes: List[KeyFrame] = []

        self._global_step = 0

    ## Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Union[Frame, SimpleFrame]) -> KeyFrame:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        else:
            raise ValueError(
                f"Can't use unknown KeyFrameSelectionStrategy {self._keyframe_selection_strategy}")

        if should_use_frame:
            if isinstance(frame, Frame):
                self._last_accepted_frame_ts = frame.start_image.timestamp
            elif isinstance(frame, SimpleFrame):
                self._last_accepted_frame_ts = frame.image.timestamp
            else:
                raise ValueError("Invalid frame type")
                
                            
            new_keyframe = KeyFrame(frame, self._device)

            # Apply the optimizated result to the new pose
            if len(self._keyframes) > 0:
                reference_kf: KeyFrame = self._keyframes[-1]

                if reference_kf._use_simple_frame:
                    tracked_reference_pose = reference_kf._tracked_lidar_pose.get_transformation_matrix().detach()
                    tracked_current_pose = new_keyframe._tracked_lidar_pose.get_transformation_matrix().detach()
                    T_track = tracked_reference_pose.inverse() @ tracked_current_pose

                    optimized_tracked_pose = reference_kf.get_lidar_pose().get_transformation_matrix().detach() @ T_track
                    new_keyframe._frame._lidar_pose = Pose(optimized_tracked_pose, requires_tensor=True)

                else:                
                    # Get tracked estimate from previous to current pose
                    tracked_reference_start_pose = reference_kf._tracked_start_lidar_pose.get_transformation_matrix().detach()
                    tracked_current_start_pose = new_keyframe._tracked_start_lidar_pose.get_transformation_matrix().detach()
                    T_track_start = tracked_reference_start_pose.inverse() @ tracked_current_start_pose

                    # Apply that transform to the optimized pose
                    start_mat = reference_kf.get_start_lidar_pose().get_transformation_matrix().detach() @ T_track_start
                    new_keyframe._frame._lidar_start_pose = Pose(start_mat, requires_tensor=True)

                    tracked_reference_end_pose = reference_kf._tracked_end_lidar_pose.get_transformation_matrix().detach()
                    tracked_current_end_pose = new_keyframe._tracked_end_lidar_pose.get_transformation_matrix().detach()
                    T_track_end = tracked_reference_end_pose.inverse() @ tracked_current_end_pose
                    end_mat = reference_kf.get_end_lidar_pose().get_transformation_matrix().detach() @ T_track_end
                    new_keyframe._frame._lidar_end_pose = Pose(end_mat, requires_tensor=True)

            if self._lidar_sample_allocation_strategy == SampleAllocationStrategy.ACTIVE:
                new_keyframe.compute_lidar_buckets()
            
            self._keyframes.append(new_keyframe)

        return new_keyframe if should_use_frame else None 

    def _select_frame_temporal(self, frame: Union[Frame, SimpleFrame]) -> bool:
        if self._last_accepted_frame_ts is None:
            return True

        if isinstance(frame, SimpleFrame):
            dt = frame.image.timestamp - self._last_accepted_frame_ts
        else:
            dt = frame.start_image.timestamp - self._last_accepted_frame_ts

        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    ## Selects which KeyFrames are to be used in the optimization, allocates
    # samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self, optimizer: Optimizer) -> List[KeyFrame]:
        window_size = self._settings.window_selection.window_size

        if self._window_selection_strategy == WindowSelectionStrategy.MOST_RECENT:
            window = self._keyframes[-window_size:]
        elif self._window_selection_strategy in [WindowSelectionStrategy.RANDOM, WindowSelectionStrategy.HYBRID]:
            
            if self._window_selection_strategy == WindowSelectionStrategy.RANDOM:
                num_temporal_frames = 1
            else:
                num_temporal_frames = self._settings.window_selection.hybrid_settings.num_recent_frames
                
            num_temporal_frames = min(num_temporal_frames, len(self._keyframes))
            indices = torch.randperm(len(self._keyframes) - num_temporal_frames)[:window_size-num_temporal_frames].tolist()

            # Note: This isn't great design, but it's pretty important that these indices comes last. 
            # Otherwise we might not keep it in the sample allocation step
            indices += list(range(-num_temporal_frames, 0))
            window = [self._keyframes[i] for i in indices]
        else:
            raise ValueError(
                f"Can't use unknown WindowSelectionStrategy {self._window_selection_strategy}")

        return self.allocate_samples(window, optimizer)
    
    ## Given the input @p keyframes, set the correct number of samples per frame. Modifies in-place.
    # @returns The updated list of keyframes.
    def allocate_samples(self, keyframes: List[KeyFrame], optimizer: Optimizer) -> List[KeyFrame]:
        uniform_rgb_samples = self._settings.sample_allocation.num_rgb_uniform_samples
        avg_rgb_samples = self._settings.sample_allocation.avg_rgb_strategy_samples

        uniform_lidar_samples = self._settings.sample_allocation.num_lidar_uniform_samples
        avg_lidar_samples = self._settings.sample_allocation.avg_lidar_strategy_samples

        num_kfs = min(self._settings.sample_allocation.num_keyframes_to_keep, len(keyframes))

        if self._window_selection_strategy == WindowSelectionStrategy.RANDOM:
            num_temporal_frames = 1
        elif self._window_selection_strategy == WindowSelectionStrategy.HYBRID:
            num_temporal_frames = self._settings.window_selection.hybrid_settings.num_recent_frames
        else:
            num_temporal_frames = 0

        num_temporal_frames = min(num_temporal_frames, len(self._keyframes))

        allocated_keyframes = []
        allocated_keyframe_idxs = []

        if self._lidar_sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
            for kf in keyframes[-num_kfs:]:
                kf.num_uniform_lidar_samples = uniform_lidar_samples
                kf.num_strategy_lidar_samples = avg_lidar_samples
                allocated_keyframes.append(kf)

        elif self._lidar_sample_allocation_strategy == SampleAllocationStrategy.ACTIVE:
            loss_distributions = []
            for kf in keyframes:
                with torch.no_grad():
                    loss_distribution = optimizer.compute_lidar_loss_distribution(kf)
                    loss_distributions.append(loss_distribution)

            loss_distributions = torch.stack(loss_distributions)
            
            if loss_distributions.dim() == 2:
                loss_distributions = loss_distributions.unsqueeze(0)

            losses = torch.sum(loss_distributions, dim=(1,2))
            losses_dist = losses / losses.sum()
            _, kept_kf_indices = losses[:-num_temporal_frames].topk(num_kfs - num_temporal_frames)

            kept_kf_indices = kept_kf_indices.tolist() + list(range(-num_temporal_frames, 0))

            total_strategy_lidar_samples = avg_lidar_samples * num_kfs
            lidar_strategy_samples = losses_dist * total_strategy_lidar_samples
            lidar_strategy_samples = lidar_strategy_samples.to(torch.int32)

            for kf_idx in kept_kf_indices:
                kf = keyframes[kf_idx]
                kf.num_uniform_lidar_samples = uniform_lidar_samples
                
                total_loss = loss_distributions[kf_idx].sum()
                kf.lidar_loss_distribution = loss_distributions[kf_idx] / total_loss
                kf.lidar_loss_distribution = kf.lidar_loss_distribution.cpu()
                
                kf.num_strategy_lidar_samples = (losses_dist[kf_idx] * total_strategy_lidar_samples).to(torch.int32)
                allocated_keyframes.append(kf)

        else:
            raise ValueError(
                f"Can't use unknown SampleAllocationStrategy {self._lidar_sample_allocation_strategy}")
  

        if self._rgb_sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
            for kf in allocated_keyframes:
                kf.num_uniform_rgb_samples = uniform_rgb_samples
                kf.num_strategy_rgb_samples = avg_rgb_samples

        elif self._rgb_sample_allocation_strategy == SampleAllocationStrategy.ACTIVE:
            loss_distributions = []
            for kf in allocated_keyframes:
                with torch.no_grad():
                    loss_distribution = optimizer.compute_rgb_loss_distribution(kf)
                loss_distributions.append(loss_distribution)

            loss_distributions = torch.stack(loss_distributions)

            losses = torch.sum(loss_distributions, dim=1)
            losses_dist = losses / losses.sum()

            total_strategy_rgb_samples = avg_rgb_samples * num_kfs
            rgb_strategy_samples = losses_dist * total_strategy_rgb_samples
            rgb_strategy_samples = rgb_strategy_samples.to(torch.int32)
            
            for kf_idx in range(len(allocated_keyframes)):
                kf: KeyFrame = allocated_keyframes[kf_idx]
                kf.num_uniform_rgb_samples = uniform_rgb_samples

                total_loss = loss_distributions[kf_idx].sum()
                kf.rgb_loss_distribution = loss_distributions[kf_idx] / total_loss
                kf.rgb_loss_distribution = kf.rgb_loss_distribution.cpu()
                
                
                kf.num_strategy_rgb_samples = (losses_dist[kf_idx] * total_strategy_rgb_samples).to(torch.int32)

                if self._settings.debug.draw_loss_distribution:
                    im_path = f"{self._settings.log_directory}/sample_alloc/allocation_{self._global_step}/keyframe_{kf_idx}.png"
                    kf.draw_loss_distribution(im_path, total_loss)

        else:
            raise ValueError(
                f"Can't use unknown SampleAllocationStrategy {self._rgb_sample_allocation_strategy}")

        self._global_step += 1
        return allocated_keyframes

    ## @returns a list of all the KeyFrame's Pose States, represented as dicts 
    def get_poses_state(self) -> List:
        result = []
        for kf in self._keyframes:
            result.append(kf.get_pose_state())
        return result