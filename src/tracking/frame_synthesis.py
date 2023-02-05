from typing import List, Union

import torch

from common.frame import Frame
from common.pose import Pose
from common.pose_utils import WorldCube
from common.sensors import Image, LidarScan
from common.settings import Settings
from tracking.sky_removal import SkyRemoval

FRAME_TOLERANCE = 0.01

class FrameSynthesis:
    """ FrameSynthesis class to process streams of data and create frames.
    """

    ## Constructor
    # @param settings: Settings for the tracker (which includes frame synthesis)
    def __init__(self, settings: Settings, T_lidar_to_camera: Pose) -> None:
        self._settings = settings

        self._sky_remover = SkyRemoval(settings.sky_removal)

        self._t_lidar_to_camera = T_lidar_to_camera
        self._t_camera_to_lidar = self._t_lidar_to_camera.inv()

        self._active_frame = Frame(T_lidar_to_camera=self._t_lidar_to_camera)

        # Frames that have both images, but might still need more lidar points
        self._in_progress_frames = []

        # Frames that are fully built and ready to be processed
        self._completed_frames = []

        # used for decimating images
        self._prev_accepted_timestamp = float('-inf')

        # Minimum dt between frames
        self._frame_delta_t_sec = 1/self._settings.frame_decimation_rate_hz
        self._frame_epislon_t_sec = self._settings.eps_t_over_delta_t * self._frame_delta_t_sec

        # This is used to queue up all the lidar points before they're assigned to frames
        self._lidar_queue = LidarScan()

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def process_lidar(self, lidar_scan: LidarScan) -> List[int]:
        self._lidar_queue.merge(lidar_scan)
        self.dequeue_lidar_points()

    # Enqueues image from @p image.
    # @precond incoming images are in monotonically increasing order (in timestamp)
    def process_image(self, image: Image, gt_pose: Pose = None) -> None:
        
        if image.timestamp - self._prev_accepted_timestamp >= self._frame_delta_t_sec - FRAME_TOLERANCE:
            self._prev_accepted_timestamp = image.timestamp

            if self._active_frame.start_image is None:
                self._active_frame.start_image = image
                if gt_pose is not None:
                    self._active_frame._gt_lidar_start_pose =  gt_pose * self._t_camera_to_lidar
            elif self._active_frame.end_image is None:
                self._active_frame.end_image = image
                if gt_pose is not None:
                    self._active_frame._gt_lidar_end_pose = gt_pose * self._t_camera_to_lidar
                self._in_progress_frames.append(self._active_frame.clone())
                self._active_frame = Frame(T_lidar_to_camera=self._t_lidar_to_camera)
                self.dequeue_lidar_points()
            else:
                raise RuntimeError("This should be unreachable")

    ## Assigns lidar points from the queue to frames
    def dequeue_lidar_points(self):
        completed_frames = 0
        for frame in self._in_progress_frames:
            start_time = frame.start_image.timestamp - self._frame_epislon_t_sec
            end_time = frame.end_image.timestamp + self._frame_epislon_t_sec

            # If there aren't lidar points to process, skip
            if len(self._lidar_queue) == 0:
                return

            # If the start time of the frame is after the last lidar point, then the
            # lidar queue is useless. Clear it out.
            if start_time > self._lidar_queue.get_end_time():
                print(
                    "Warning: Got an image that starts after all queued lidar points. Dropping lidar points")
                self._lidar_queue.clear()
                return

            # If the end time of the frame comes before all of the lidar points,
            # then the frame is assumed done. Mark it complete and move on.
            if end_time < self._lidar_queue.get_start_time():
                completed_frames += 1
                continue

            # If there are some points we need to skip
            if self._lidar_queue.get_start_time() < start_time:
                first_valid_idx = torch.argmax(
                    (self._lidar_queue.timestamps >= start_time).float())
            else:
                first_valid_idx = 0

            # It's possible some of the input points should actually be added to future scans
            if self._lidar_queue.get_end_time() > end_time:
                last_valid_idx = torch.argmax(
                    (self._lidar_queue.timestamps > end_time).float())
            else:
                last_valid_idx = len(self._lidar_queue)

            # Get the points from the queue that need to be moved to the current frame
            new_ray_directions = self._lidar_queue.ray_directions[..., first_valid_idx:last_valid_idx:10]
            new_distances = self._lidar_queue.distances[first_valid_idx:last_valid_idx:10]
            new_timestamps = self._lidar_queue.timestamps[first_valid_idx:last_valid_idx:10]

            single_origin = self._lidar_queue.ray_origin_offsets.dim() == 2
            if single_origin:
                new_ray_origins = self._lidar_queue.ray_origin_offsets
            else:
                new_ray_origins = self._lidar_queue.ray_origin_offsets[...,
                                                                       first_valid_idx:last_valid_idx:10]

            frame.lidar_points.add_points(
                new_ray_directions, new_distances, new_ray_origins, new_timestamps)

            self._lidar_queue.remove_points(last_valid_idx)

            # If any points remain in the queue, then the frame must be done since the remaining points
            # imply the existance of points that come after the frame.
            if len(self._lidar_queue) > 0:
                completed_frames += 1

        # For all the frames that are now complete, bundle them up and send off for tracking
        for _ in range(completed_frames):
            frame = self._in_progress_frames.pop(0)
            frame.set_start_sky_mask(
                self._sky_remover.get_sky_mask(frame.start_image))
            frame.set_end_sky_mask(
                self._sky_remover.get_sky_mask(frame.end_image))
            self._completed_frames.append(frame)

    ## Check if a frame exists to be returned
    def has_frame(self) -> bool:
        return len(self._completed_frames) != 0

    ## Return and remove a newly synthesized Frame. If unavailable, returns None.
    def pop_frame(self) -> Union[Frame, None]:

        # note: this is done with queues to avoid potentially expensive copies
        # which would be needed to avoid active_frame getting overwritten.
        if len(self._completed_frames) == 0:
            return None

        return self._completed_frames.pop(0)
