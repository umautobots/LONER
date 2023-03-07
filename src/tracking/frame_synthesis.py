import enum
from typing import List, Union

import torch

from common.frame import Frame, SimpleFrame
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

        self._use_simple_frames = self._settings.use_simple_frames
        self._split_lidar_scans = self._settings.split_lidar_scans

        assert not(not self._use_simple_frames and self._split_lidar_scans), \
                "split_lidar_scans can't be True if use_simple_frames is False"

        self._lidar_scans = []
        self._lidar_scan_timestamps = torch.Tensor()

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def process_lidar(self, lidar_scan: LidarScan) -> List[int]:
        if self._split_lidar_scans:
            self._lidar_queue.merge(lidar_scan)
        else:
            self._lidar_scans.append(lidar_scan)
            scan_ts = torch.Tensor([lidar_scan.timestamps[0], lidar_scan.timestamps[-1]]).view(2,1)
            self._lidar_scan_timestamps = torch.hstack((self._lidar_scan_timestamps, scan_ts))

        self.create_frames()

    # Enqueues image from @p image.
    # @precond incoming images are in monotonically increasing order (in timestamp)
    def process_image(self, image: Image, gt_pose: Pose = None) -> None:
        if self._use_simple_frames:
            if image.timestamp - self._prev_accepted_timestamp >= self._frame_delta_t_sec - FRAME_TOLERANCE:
                self._prev_accepted_timestamp = image.timestamp
                new_frame = SimpleFrame(image=image, T_lidar_to_camera=self._t_lidar_to_camera)
                if gt_pose is not None:
                    new_frame._gt_lidar_pose = gt_pose * self._t_camera_to_lidar
                    
                self._in_progress_frames.append(new_frame.clone())
                self.create_frames()
        else:
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
                    self.create_frames()
                else:
                    raise RuntimeError("This should be unreachable")

    def create_frames(self):
        if self._split_lidar_scans:
            self._dequeue_lidar_points()
        else:
            self._match_images_to_scans()

    ## Assigns lidar points from the queue to frames
    def _dequeue_lidar_points(self):
        completed_frames = 0
        for frame in self._in_progress_frames:
            if self._use_simple_frames:
                start_time = frame.image.timestamp - self._frame_delta_t_sec / 2
                end_time = frame.image.timestamp + self._frame_delta_t_sec / 2
            else:
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

            point_step = self._settings.lidar_point_step

            # Get the points from the queue that need to be moved to the current frame
            new_ray_directions = self._lidar_queue.ray_directions[..., first_valid_idx:last_valid_idx:point_step]
            new_distances = self._lidar_queue.distances[first_valid_idx:last_valid_idx:point_step]
            new_timestamps = self._lidar_queue.timestamps[first_valid_idx:last_valid_idx:point_step]


            frame.lidar_points.add_points(
                new_ray_directions, new_distances, new_timestamps)

            self._lidar_queue.remove_points(last_valid_idx)

            # If any points remain in the queue, then the frame must be done since the remaining points
            # imply the existance of points that come after the frame.
            if len(self._lidar_queue) > 0:
                completed_frames += 1

        # For all the frames that are now complete, bundle them up and send off for tracking
        for _ in range(completed_frames):
            frame = self._in_progress_frames.pop(0)
            if len(frame.lidar_points) == 0:
                continue
            self._completed_frames.append(frame)

    def _match_images_to_scans(self):

        """
        
        images: 0.1

        scans: 0.0->0.1, 0.1->0.2, 0.2->0.3
        
        lidar_start_times = [-0.01, 0.09, 0.19]
        lidar_end_times = [0.11, 0.21, 0.31]

        valid_start = [True True False]
        valid_end = [True True True]

        valid_scans = [True True False]

        chosen_scan_idx = 0
        
        """

        class MatchResult(enum.Enum):
            COMPLETED = 0,
            SKIPPED = 1

        results = []
        
        for frame in self._in_progress_frames:

            if len(self._lidar_scans) == 0:
                break

            lidar_start_times = self._lidar_scan_timestamps[0] - self._settings.frame_match_tolerance
            lidar_end_times = self._lidar_scan_timestamps[1] + self._settings.frame_match_tolerance

            if lidar_start_times[0] > frame.image.timestamp:
                results.append(MatchResult.SKIPPED)

            valid_start = frame.image.timestamp >= lidar_start_times
            valid_end = frame.image.timestamp <= lidar_end_times

            chosen_scan_idx = torch.argmax(torch.bitwise_and(valid_start, valid_end).float())

            frame.lidar_points = self._lidar_scans[chosen_scan_idx]

            self._lidar_scans = self._lidar_scans[chosen_scan_idx+1:]
            self._lidar_scan_timestamps = self._lidar_scan_timestamps[:, chosen_scan_idx+1:]

            results.append(MatchResult.COMPLETED)

        for result in results:
            frame: Union[Frame, SimpleFrame] = self._in_progress_frames.pop(0)
            if result == MatchResult.COMPLETED:
                if len(frame.lidar_points) == 0:
                    continue
                self._completed_frames.append(frame)
            else:
                print("Skipped frame with image TS", frame.image.timestamp)

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
