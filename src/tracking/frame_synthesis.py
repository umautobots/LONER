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

        # Frames that are in-progress - might need more images or lidar points.
        # Should always have at least one entry - even if it's empty.
        # Only the most recent entry can need images. 
        # The others can be waiting for up-to-date lidar poitns. A frame isn't 
        # closed until a lidar point that comes after its end time is recieved.
        self._in_progress_frames = [Frame(T_lidar_to_camera=self._t_lidar_to_camera)]

        # Frames that are fully built and ready to be processed
        self._completed_frames = []

        self._prev_accepted_timestamp = float('-inf')

        self._frame_delta_t_sec = 1/self._settings.frame_decimation_rate_hz

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def ProcessLidar(self, lidar_scan: LidarScan) -> List[int]:

        num_completed_frames = 0
        for frame_idx in range(len(self._in_progress_frames)):
            
            if self._in_progress_frames[frame_idx].end_image is None:
                break

            start_time = self._in_progress_frames[frame_idx].start_image.timestamp - self._frame_delta_t_sec/2
            end_time = self._in_progress_frames[frame_idx].end_image.timestamp + self._frame_delta_t_sec/2

            if lidar_scan.timestamps[0] < start_time:
                print("Warning: Got Lidar points for an already completed frame. Skipping")
                first_valid_idx = torch.argmax((lidar_scan.timestamps >= start_time).float())
            else:
                first_valid_idx = 0

            # It's possible some of the input points should actually be added to future scans
            if lidar_scan.timestamps[-1] > end_time:
                last_valid_idx = torch.argmax((lidar_scan.timestamps > end_time).float())
            else:
                last_valid_idx = len(lidar_scan.timestamps)

            new_ray_directions = lidar_scan.ray_directions[first_valid_idx:last_valid_idx, ...]
            new_timestamps = lidar_scan.timestamps[first_valid_idx:last_valid_idx]

            if lidar_scan.ray_origin_offsets.dim() == 2:
                new_ray_origins = lidar_scan.ray_origin_offsets
            else:
                new_ray_origins = lidar_scan.ray_origin_offsets[first_valid_idx:last_valid_idx, ...]
            
            self._in_progress_frames[frame_idx].lidar_points.AddPoints(new_ray_directions,
                                                                      new_ray_origins,
                                                                      new_timestamps)

            num_completed_frames += 1

            if last_valid_idx == len(lidar_scan.timestamps):
                break

        for _ in range(num_completed_frames):
            frame = self._in_progress_frames.pop(0)
            frame.SetStartSkyMask(self._sky_remover.GetSkyMask(frame.start_image))
            frame.SetEndSkyMask(self._sky_remover.GetSkyMask(frame.end_image))
            self._completed_frames.append(frame)

    ## Enqueues image from @p image.
    # @precond incoming images are in monotonically increasing order (in timestamp)
    def ProcessImage(self, image: Image) -> None:
        
        if image.timestamp - self._prev_accepted_timestamp >= self._frame_delta_t_sec:
            self._prev_accepted_timestamp = image.timestamp

            most_recent_frame = self._in_progress_frames[-1]

            if most_recent_frame.start_image is None:
                most_recent_frame.start_image = image
            elif most_recent_frame.end_image is None:
                most_recent_frame.end_image = image
                self._in_progress_frames.append(Frame(T_lidar_to_camera=self._t_lidar_to_camera))
            else:
                raise RuntimeError("This should be unreachable")

    def HasFrame(self) -> bool:
        return len(self._completed_frames) != 0

    ## Return and remove a newly synthesized Frame. If unavailable, returns None.
    def PopFrame(self) -> Union[Frame, None]:

        # note: this is done with queues to avoid potentially expensive copies
        # which would be needed to avoid active_frame getting overwritten.
        if len(self._completed_frames) == 0:
            return None
        return self._completed_frames.pop(0)