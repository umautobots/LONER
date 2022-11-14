import copy
import torch
from typing import List, Union

from common.frame import Frame
from common.sensors import Image, LidarScan
from common.settings import Settings
from common.pose_utils import Pose

from tracking.sky_removal import SkyRemoval


class FrameSynthesis:
    """ FrameSynthesis class to process streams of data and create frames.
    """

    ## Constructor
    # @param settings: Settings for the tracker (which includes frame synthesis)
    def __init__(self, settings: Settings, T_lidar_to_camera: Pose) -> None:
        self._settings = settings

        self._sky_remover = SkyRemoval(settings.sky_removal)

        self._t_lidar_to_camera = T_lidar_to_camera

        self._active_frame = Frame()

        # Frames that have both images, but might still need more lidar points
        self._in_progress_frames = []

        # Frames that are fully built and ready to be processed
        self._completed_frames = []

        # used for decimating images
        self._prev_accepted_timestamp = float('-inf')

        # Minimum dt between frames
        self._frame_delta_t_sec = 1/self._settings.frame_decimation_rate_hz

        # This is used to queue up all the lidar points before they're assigned to frames
        self._lidar_queue = LidarScan()

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def ProcessLidar(self, lidar_scan: LidarScan) -> List[int]:
        self._lidar_queue.Merge(lidar_scan)
        self.DequeueLidarPoints()

    ## Enqueues image from @p image.
    # @precond incoming images are in monotonically increasing order (in timestamp)
    def ProcessImage(self, image: Image) -> None:
        if image.timestamp - self._prev_accepted_timestamp >= self._frame_delta_t_sec:
            self._prev_accepted_timestamp = image.timestamp

            if self._active_frame.start_image is None:
                self._active_frame.start_image = image
            elif self._active_frame.end_image is None:
                self._active_frame.end_image = image
                self._in_progress_frames.append(copy.deepcopy(self._active_frame))
                self._active_frame = Frame()
                self.DequeueLidarPoints()
            else:
                raise RuntimeError("This should be unreachable")

    ## Assigns lidar points from the queue to frames
    def DequeueLidarPoints(self):
        completed_frames = 0
        for frame in self._in_progress_frames:
            start_time = frame.start_image.timestamp - self._frame_delta_t_sec/2
            end_time = frame.end_image.timestamp + self._frame_delta_t_sec/2
            
            # If there aren't lidar points to process, skip
            if len(self._lidar_queue) == 0:
                return

            # If the start time of the frame is after the last lidar point, then the
            # lidar queue is useless. Clear it out.
            if start_time > self._lidar_queue.GetEndTime():
                print("Warning: Got an image that starts after all queued lidar points. Dropping lidar points")
                self._lidar_queue.Clear()
                return

            # If the end time of the frame comes before all of the lidar points,
            # then the frame is assumed done. Mark it complete and move on.
            if end_time < self._lidar_queue.GetStartTime():
                completed_frames += 1
                continue
            
            # If there are some points we need to skip
            if self._lidar_queue.GetStartTime() < start_time:
                print("Warning: Got Lidar points for an already completed frame. Skipping")
                first_valid_idx = torch.argmax((self._lidar_queue.timestamps >= start_time).float())
            else:
                first_valid_idx = 0

            # It's possible some of the input points should actually be added to future scans
            if self._lidar_queue.GetEndTime() > end_time:
                last_valid_idx = torch.argmax((self._lidar_queue.timestamps > end_time).float())
            else:
                last_valid_idx = len(self._lidar_queue)

            # Get the points from the queue that need to be moved to the current frame
            new_ray_directions = self._lidar_queue.ray_directions[first_valid_idx:last_valid_idx]
            new_distances = self._lidar_queue.distances[first_valid_idx:last_valid_idx]
            new_timestamps = self._lidar_queue.timestamps[first_valid_idx:last_valid_idx]
            
            single_origin = self._lidar_queue.ray_origin_offsets.dim() == 2
            if single_origin:
                new_ray_origins = self._lidar_queue.ray_origin_offsets
            else:
                new_ray_origins = self._lidar_queue.ray_origin_offsets[..., first_valid_idx:last_valid_idx]

            frame.lidar_points.AddPoints(new_ray_directions, new_distances, new_ray_origins, new_timestamps)
            

            self._lidar_queue.RemovePoints(last_valid_idx)

            # If any points remain in the queue, then the frame must be done since the remaining points
            # imply the existance of points that come after the frame.
            if len(self._lidar_queue) > 0:
                completed_frames += 1
        
        # For all the frames that are now complete, bundle them up and send off for tracking
        for _ in range(completed_frames):
            frame = self._in_progress_frames.pop(0)
            frame.SetStartSkyMask(self._sky_remover.GetSkyMask(frame.start_image))
            frame.SetEndSkyMask(self._sky_remover.GetSkyMask(frame.end_image))
            self._completed_frames.append(frame)

    def HasFrame(self) -> bool:
        return len(self._completed_frames) != 0

    ## Return and remove a newly synthesized Frame. If unavailable, returns None.
    def PopFrame(self) -> Union[Frame, None]:

        # note: this is done with queues to avoid potentially expensive copies
        # which would be needed to avoid active_frame getting overwritten.
        if len(self._completed_frames) == 0:
            return None
        return self._completed_frames.pop(0)