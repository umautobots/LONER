import torch
import torch.multiprocessing as mp
import os
import numpy as np
from common.frame import Frame
from common.settings import Settings
from common.utils import StopSignal
from tracking.frame_synthesis import FrameSynthesis
from common.signals import Slot, Signal
from common.pose_utils import Pose
import pytorch3d
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# Yanked from http://www.open3d.org/docs/release/python_example/pipelines/index.html#icp-registration-py
def transform_cloud(source, transformation):

    source_temp = o3d.cuda.pybind.geometry.PointCloud()

    source_temp.points = copy.deepcopy(source.points)

    source_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transformation)
    
    return source_temp

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

        # Used for frame-to-frame ICP tracking
        self._reference_point_cloud = None
        self._reference_pose = Pose(fixed=True)
        self._reference_time = None

        self._frame_count = 0

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def Run(self) -> None:
        while True:            
            if self._rgb_slot.HasValue():
                new_rgb = self._rgb_slot.GetValue()

                if isinstance(new_rgb, StopSignal):
                    break

                self._frame_synthesizer.ProcessImage(new_rgb)

            if self._lidar_slot.HasValue():
                new_lidar = self._lidar_slot.GetValue()

                if isinstance(new_lidar, StopSignal):
                    break
                
                self._frame_synthesizer.ProcessLidar(new_lidar)

            while self._frame_synthesizer.HasFrame():
                print("Tracking Frame")
                frame = self._frame_synthesizer.PopFrame()
                tracked = self.TrackFrame(frame)
                print("end pose:", frame._lidar_end_pose)
                if not tracked:
                    print("Warning: Failed to track frame. Skipping.")
                    continue
                self._frame_signal.Emit(frame)
        
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
        
        # First Iteration: No reference pose, we fix this as the origin of the coordinate system.
        if self._reference_point_cloud is None:
            frame._lidar_start_pose = self._reference_pose.Clone(fixed=True)
            frame._lidar_end_pose = self._reference_pose.Clone(fixed=True)
            self._reference_point_cloud = frame.BuildPointCloud()
            self._reference_time = (frame.start_image.timestamp + frame.end_image.timestamp) / 2
            return True

        device = self._reference_pose.GetTransformationMatrix().device

        # Future Iterations: Actually do ICP
        frame_point_cloud = frame.BuildPointCloud()

        initial_guess = self._reference_pose.GetTransformationMatrix().cpu().numpy()
                
        print("Source Size:", len(self._reference_point_cloud.points), "Target Size:", len(frame_point_cloud.points))
        
        # TODO: See the estimate normals section of http://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#
        # They recommend doing it a different way by setting a radius, but I don't know what radius to use
        self._reference_point_cloud.estimate_normals()
        frame_point_cloud.estimate_normals()
        
        convergence_criteria = o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
            self._settings.icp.relative_fitness, self._settings.icp.relative_rmse, self._settings.icp.max_iterations)

        registration = o3d.pipelines.registration.registration_icp(
            self._reference_point_cloud, frame_point_cloud, self._settings.icp.threshold, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria=convergence_criteria)

        registration_result = torch.from_numpy(registration.transformation.copy()).float().to(device)

        tracked_position = self._reference_pose.GetTransformationMatrix() @ registration_result

        # Do interpolation/extrapolation to get the start/end lidar poses
        # TODO: Is there a better way to do this extrapolation? 
        rot = R.from_matrix(registration_result[:3, :3])
        rot_vec = rot.as_rotvec()
        rot_amount = np.linalg.norm(rot_vec)
        rot_axis = rot_vec / rot_amount
        trans_vec = tracked_position[:3, 3]

        new_reference_time = (frame.start_image.timestamp + frame.end_image.timestamp) / 2
        reference_pose_mat = self._reference_pose.GetTransformationMatrix()

        start_time_interp_factor = (frame.start_image.timestamp - self._reference_time)/(new_reference_time - self._reference_time)
        rot_vec_start = rot_axis * start_time_interp_factor * rot_amount
        rot_start = R.from_rotvec(rot_vec_start)
        trans_vec_start = reference_pose_mat[:3, 3] + start_time_interp_factor * trans_vec
        start_transformation_mat = np.hstack((rot_start.as_matrix(), trans_vec_start.reshape(3,1)))
        start_transformation_mat = np.vstack((start_transformation_mat, [0,0,0,1]))
        frame._lidar_start_pose = Pose(torch.from_numpy(start_transformation_mat).float().to(device))

        end_time_interp_factor = (frame.end_image.timestamp - self._reference_time)/(new_reference_time - self._reference_time)
        rot_vec_end = rot_axis * end_time_interp_factor * rot_amount
        rot_end = R.from_rotvec(rot_vec_end)
        trans_vec_end = reference_pose_mat[:3, 3] + end_time_interp_factor * trans_vec
        end_transformation_mat = np.hstack((rot_end.as_matrix(), trans_vec_end.reshape(3,1)))
        end_transformation_mat = np.vstack((end_transformation_mat, [0,0,0,1]))
        frame._lidar_end_pose = Pose(torch.from_numpy(end_transformation_mat).float().to(device))
        print("Set End Pose To:", frame._lidar_end_pose)

        logdir = f"../outputs/frame_{self._frame_count}"
        os.mkdir(f"../outputs/frame_{self._frame_count}")
        o3d.io.write_point_cloud(f"{logdir}/reference_point_cloud.pcd", self._reference_point_cloud)
        o3d.io.write_point_cloud(f"{logdir}/frame_point_cloud.pcd", frame_point_cloud)
        o3d.io.write_point_cloud(f"{logdir}/transfromed_reference_cloud.pcd", transform_cloud(self._reference_point_cloud, registration_result))
        

        self._reference_pose = Pose(tracked_position, fixed=True)
        self._reference_point_cloud = frame_point_cloud

        # print(f"Frame {self._frame_count} registration_result:", registration_result)
        
        
        self._frame_count += 1
        return True