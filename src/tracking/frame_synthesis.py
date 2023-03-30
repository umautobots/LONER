import enum
from typing import List, Union

import torch

from common.frame import Frame
from common.pose import Pose
from common.pose_utils import WorldCube
from common.sensors import Image, LidarScan
from common.settings import Settings

class FrameSynthesis:
    """ FrameSynthesis class to process streams of data and create frames.
    """

    ## Constructor
    # @param settings: Settings for the tracker (which includes frame synthesis)
    def __init__(self, settings: Settings, T_lidar_to_camera: Pose, lidar_only: bool) -> None:
        self._settings = settings

        self._t_lidar_to_camera = T_lidar_to_camera
        self._t_camera_to_lidar = self._t_lidar_to_camera.inv()

        self._lidar_only = lidar_only

        self._active_frame = Frame(T_lidar_to_camera=self._t_lidar_to_camera)

        # Frames that have both images, but might still need more lidar points
        self._in_progress_frames = []

        # Frames that are fully built and ready to be processed
        self._completed_frames = []

        # used for decimating images
        self._prev_accepted_timestamp = float('-inf')

        # Minimum dt between frames
        self._frame_delta_t_sec = 1/self._settings.frame_decimation_rate_hz

        self._lidar_scans = []
        self._lidar_scan_timestamps = torch.Tensor()

        self._decimate_on_load = self._settings.decimate_on_load

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def process_lidar(self, lidar_scan: LidarScan, gt_pose: Pose) -> List[int]:
        if self._lidar_only:
            scan_time = lidar_scan.get_start_time()
            dt = self._frame_delta_t_sec - self._settings.frame_delta_t_sec_tolerance
            if self._decimate_on_load or scan_time - self._prev_accepted_timestamp >= dt:
                new_frame = Frame(None, lidar_scan, self._t_lidar_to_camera)
                new_frame._gt_lidar_pose = gt_pose
                self._completed_frames.append(new_frame.clone())
                self._prev_accepted_timestamp = lidar_scan.get_start_time()
        else:
            self._lidar_scans.append((lidar_scan, gt_pose))
            scan_ts = torch.Tensor([lidar_scan.timestamps[0], lidar_scan.timestamps[-1]]).view(2,1)
            self._lidar_scan_timestamps = torch.hstack((self._lidar_scan_timestamps, scan_ts))
            self.create_frames()

    # Enqueues image from @p image.
    # @precond incoming images are in monotonically increasing order (in timestamp)
    def process_image(self, image: Image) -> None:
        if image.timestamp - self._prev_accepted_timestamp >= self._frame_delta_t_sec - self._settings.frame_delta_t_sec_tolerance:
            self._prev_accepted_timestamp = image.timestamp
            new_frame = Frame(image=image, T_lidar_to_camera=self._t_lidar_to_camera)

            self._in_progress_frames.append(new_frame.clone())
            self.create_frames()

    def create_frames(self):
        self._match_images_to_scans()

    def _match_images_to_scans(self):

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
                print(lidar_start_times[0], frame.image.timestamp)
                results.append(MatchResult.SKIPPED)

            valid_start = frame.image.timestamp >= lidar_start_times
            valid_end = frame.image.timestamp <= lidar_end_times

            chosen_scan_idx = torch.argmax(torch.bitwise_and(valid_start, valid_end).float())

            frame.lidar_points = self._lidar_scans[chosen_scan_idx][0]
            frame._gt_lidar_pose = self._lidar_scans[chosen_scan_idx][1]

            self._lidar_scans = self._lidar_scans[chosen_scan_idx+1:]
            self._lidar_scan_timestamps = self._lidar_scan_timestamps[:, chosen_scan_idx+1:]

            results.append(MatchResult.COMPLETED)

        for result in results:
            frame: Frame = self._in_progress_frames.pop(0)
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
