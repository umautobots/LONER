import torch.multiprocessing as mp
from common.frame import Frame
from common.settings import Settings
from common.utils import StopSignal
from tracking.frame_synthesis import FrameSynthesis
from common.signals import Slot, Signal
from common.pose_utils import Pose


class Tracker:
    """ Tracker: Top-level Tracking module
    
    Given streams of RGB data and lidar scans, this class is responsible for creating
    Frame instances and estimating their poses.
    """

    ## Constructor
    # @param settings: Top level settings for the entire Cloner-SLAM module. Needed for calib etc.
    # @param rgb_signal: A Signal which the Tracker creates a Slot for, used for fetching RGB frames
    # @param lidar_signal: Same as rgb_signal, but for lidar
    # @param frame_queue: A Signal which the Tracker emits to when it completes a frame
    def __init__(self,
                 settings: Settings,
                 rgb_signal: Signal,
                 lidar_signal: Signal,
                 frame_signal: Signal) -> None:

        self._rgb_slot = rgb_signal.Register()
        self._lidar_slot = lidar_signal.Register()
        self._frame_signal = frame_signal
        self._settings = settings.tracker
        
        self._t_lidar_to_cam = Pose.FromSettings(settings.calibration.lidar_to_camera)

        self._frame_synthesizer = FrameSynthesis(self._settings.frame_synthesis, self._t_lidar_to_cam)

        # Used to indicate to an external process that I've processed the stop signal
        self._processed_stop_signal = mp.Value('i', 0)
        # Set to 0 from an external thread when it's time to actuall exit.
        self._term_signal = mp.Value('i', 0)

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def Run(self) -> None:
        while True:            
            if self._rgb_slot.HasValue():
                new_rgb = self._rgb_slot.GetValue()

                if isinstance(new_rgb, StopSignal):
                    break

                print("Tracking rgb frame", new_rgb.timestamp)

                self._frame_synthesizer.ProcessImage(new_rgb)

            if self._lidar_slot.HasValue():
                new_lidar = self._lidar_slot.GetValue()

                if isinstance(new_lidar, StopSignal):
                    break
                
                self._frame_synthesizer.ProcessLidar(new_lidar)

            while self._frame_synthesizer.HasFrame():
                self._frame_signal.Emit(self._frame_synthesizer.PopFrame())
        
        self._processed_stop_signal.value = True

        print("Tracking Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting tracking process.")
        
    ## TrackFrame inputs a @p frame and estimates its pose, which is stored in the Frame.
    # @returns True if tracking was successful.
    def TrackFrame(self, frame: Frame) -> bool:
        pass