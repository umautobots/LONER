from typing import List
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

class SampleAllocationStrategy(Enum):
    UNIFORM = 0
    ACTIVE = 1

class KeyFrameManager:
    """ The KeyFrame Manager class creates and manages KeyFrames and passes 
    data to the optimizer.
    """

    # Constructor
    # @param settings: Settings object for the KeyFrame Manager
    def __init__(self, settings: Settings, device: int = 'cpu') -> None:
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

        self._global_step = 0

    # Processes the input @p frame, decides whether it's a KeyFrame, and if so
    # adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame) -> KeyFrame:
        if self._keyframe_selection_strategy == KeyFrameSelectionStrategy.TEMPORAL:
            should_use_frame = self._select_frame_temporal(frame)
        else:
            raise ValueError(
                f"Can't use unknown KeyFrameSelectionStrategy {self._keyframe_selection_strategy}")

        if should_use_frame:
            self._last_accepted_frame_ts = frame.start_image.timestamp
            new_keyframe = KeyFrame(frame, self._device)

            # Apply the optimizated result to the new pose
            if len(self._keyframes) > 0:
                reference_kf: KeyFrame = self._keyframes[-1]

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

            self._keyframes.append(new_keyframe)

        return new_keyframe if should_use_frame else None 

    def _select_frame_temporal(self, frame: Frame) -> bool:
        if self._last_accepted_frame_ts is None:
            return True

        dt = frame.start_image.timestamp - self._last_accepted_frame_ts
        dt_threshold = self._settings.keyframe_selection.temporal.time_diff_seconds
        return dt >= dt_threshold

    # Selects which KeyFrames are to be used in the optimization, allocates
    # samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self, optimizer: Optimizer) -> List[KeyFrame]:
        window_size = self._settings.window_selection.window_size

        if self._window_selection_strategy == WindowSelectionStrategy.MOST_RECENT:
            window = self._keyframes[-window_size:]
        elif self._window_selection_strategy == WindowSelectionStrategy.RANDOM:
            indices = torch.randperm(len(self._keyframes) - 1)[:window_size-1].tolist()

            # Note: This isn't great design, but it's pretty important that the -1 index comes last. 
            # Otherwise we might not keep it in the sample allocation step
            indices.append(-1)
            window = [self._keyframes[i] for i in indices]
        else:
            raise ValueError(
                f"Can't use unknown WindowSelectionStrategy {self._window_selection_strategy}")

        return self.allocate_samples(window, optimizer)
    
    # Given the input @p keyframes, set the correct number of samples per frame. Modifies in-place.
    # @returns The updated list of keyframes.
    def allocate_samples(self, keyframes: List[KeyFrame], optimizer: Optimizer) -> List[KeyFrame]:
        uniform_rgb_samples = self._settings.sample_allocation.num_rgb_uniform_samples
        avg_rgb_samples = self._settings.sample_allocation.avg_rgb_strategy_samples

        uniform_lidar_samples = self._settings.sample_allocation.num_lidar_uniform_samples
        avg_lidar_samples = self._settings.sample_allocation.avg_lidar_strategy_samples

        num_kfs = min(self._settings.sample_allocation.num_keyframes_to_keep, len(keyframes))

        allocated_keyframes = []
        if self._sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
            for kf in keyframes[-num_kfs:]:
                kf.num_uniform_rgb_samples = uniform_rgb_samples
                kf.num_strategy_rgb_samples = avg_rgb_samples
                kf.num_uniform_lidar_samples = uniform_lidar_samples
                kf.num_strategy_lidar_samples = avg_lidar_samples
                allocated_keyframes.append(kf)
        elif self._sample_allocation_strategy == SampleAllocationStrategy.ACTIVE:
            loss_distributions = []
            for kf in keyframes:
                with torch.no_grad():
                    loss_distribution = optimizer.compute_rgb_loss_distribution(kf)
                loss_distributions.append(loss_distribution)

            loss_distributions = torch.stack(loss_distributions)

            losses = torch.sum(loss_distributions, dim=1)
            _, kept_kf_indices = losses.topk(num_kfs)

            # Make sure we keep the most recent.
            if len(losses) - 1 not in kept_kf_indices:
                kept_kf_indices[-1] = len(losses) - 1

            print(kept_kf_indices, len(losses))
            print([keyframes[i].get_start_time() for i in kept_kf_indices])
            losses_dist = losses / losses.sum()

            total_strategy_rgb_samples = avg_rgb_samples * len(kept_kf_indices)
            rgb_strategy_samples = losses_dist * total_strategy_rgb_samples
            rgb_strategy_samples = rgb_strategy_samples.to(torch.int32)
            
            if self._settings.debug.draw_loss_distribution:
                kfs_to_compute = torch.arange(len(keyframes))
            else:
                kfs_to_compute = kept_kf_indices

            for kf_idx in kfs_to_compute:
                kf = keyframes[kf_idx]
                kf.num_uniform_lidar_samples = uniform_lidar_samples
                kf.num_uniform_rgb_samples = uniform_rgb_samples

                total_loss = loss_distributions[kf_idx].sum()
                kf.loss_distribution = loss_distributions[kf_idx] / total_loss
                kf.loss_distribution = kf.loss_distribution.cpu()

                kf.num_strategy_lidar_samples = avg_lidar_samples
                
                kf.num_strategy_rgb_samples = (losses_dist[kf_idx] * total_strategy_rgb_samples).to(torch.int32)

                if self._settings.debug.draw_loss_distribution:
                    im_path = f"{self._settings.log_directory}/sample_alloc/allocation_{self._global_step}/keyframe_{kf_idx}.png"
                    kept = kf_idx in kept_kf_indices.cpu()
                    kf.draw_loss_distribution(im_path, total_loss, kept)

                    if kept:
                        allocated_keyframes.append(kf)
                else:
                    allocated_keyframes.append(kf)

        else:
            raise ValueError(
                f"Can't use unknown SampleAllocationStrategy {self._sample_allocation_strategy}")

        self._global_step += 1
        return allocated_keyframes

    def get_poses_state(self) -> List:
        result = []
        for kf in self._keyframes:
            result.append(kf.get_pose_state())
        return result