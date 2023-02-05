import cProfile
import copy
import os

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from common.frame import Frame
from common.pose import Pose
from common.pose_utils import WorldCube
from common.settings import Settings
from common.signals import Signal, StopSignal
from tracking.frame_synthesis import FrameSynthesis


# Yanked from http://www.open3d.org/docs/release/python_example/pipelines/index.html#icp-registration-py
def transform_cloud(source, transformation):

    source_temp = o3d.cuda.pybind.geometry.PointCloud()

    source_temp.points = copy.deepcopy(source.points)

    source_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transformation)

    return source_temp


class Tracker:
    """Tracker: Top-level Tracking module

    Given streams of RGB data and lidar scans, this class is responsible for creating
    Frame instances and estimating their poses.
    """

    ## Constructor
    # @param settings: Top level settings for the entire Cloner-SLAM module. Needed for calib etc.
    # @param rgb_signal: A Signal which the Tracker creates a Slot for, used for fetching RGB frames
    # @param lidar_signal: Same as rgb_signal, but for lidar
    # @param frame_queue: A Signal which the Tracker emits to when it completes a frame
    def __init__(
            self,
            settings: Settings,
            rgb_signal: Signal,
            lidar_signal: Signal,
            frame_signal: Signal) -> None:

        self._rgb_slot = rgb_signal.register()
        self._lidar_slot = lidar_signal.register()
        self._frame_signal = frame_signal
        self._settings = settings.tracker

        self._t_lidar_to_camera = Pose.from_settings(
            settings.calibration.lidar_to_camera)

        self._frame_synthesizer = FrameSynthesis(
            self._settings.frame_synthesis, self._t_lidar_to_camera)

        # Used to indicate to an external process that I've processed the stop signal
        self._processed_stop_signal = mp.Value("i", 0)
        # Set to 0 from an external thread when it's time to actuall exit.
        self._term_signal = mp.Value("i", 0)

        # Used for frame-to-frame ICP tracking
        self._reference_point_cloud = None
        self._reference_pose = Pose(fixed=True)
        self._reference_time = None

        self._frame_count = 0

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def run(self) -> None:
        while True:
            if self._rgb_slot.has_value():
                val = self._rgb_slot.get_value()

                if isinstance(val, StopSignal):
                    break

                new_rgb, new_gt_pose = val

                self._frame_synthesizer.process_image(new_rgb, new_gt_pose)

            if self._lidar_slot.has_value():
                new_lidar = self._lidar_slot.get_value()

                if isinstance(new_lidar, StopSignal):
                    break

                self._frame_synthesizer.process_lidar(new_lidar)

            while self._frame_synthesizer.has_frame():
                frame = self._frame_synthesizer.pop_frame()
                tracked = self.track_frame(frame)

                if not tracked:
                    print("Warning: Failed to track frame. Skipping.")
                    continue
                
                if self._settings.debug.write_frame_point_clouds:
                    pcd = frame.build_point_cloud()
                    logdir = f"{self._settings.log_directory}/frames/"
                    os.makedirs(logdir, exist_ok=True)
                    o3d.io.write_point_cloud(
                        f"{logdir}/cloud_{self._frame_count}.pcd", pcd)

                self._frame_signal.emit(frame)

        self._processed_stop_signal.value = True

        print("Tracking Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting tracking process.")

    ## track_frame inputs a @p frame and estimates its pose, which is stored in the Frame.
    # @returns True if tracking was successful.
    def track_frame(self, frame: Frame) -> bool:

        downsample_type = self._settings.icp.downsample.type

        if downsample_type is None:
            frame_point_cloud = frame.build_point_cloud(0.09)
        elif downsample_type == "VOXEL":
            frame_point_cloud = frame.build_point_cloud(0.09)
            voxel_size = self._settings.icp.downsample.voxel_downsample_size
            frame_point_cloud = frame_point_cloud.voxel_down_sample(
                voxel_size=voxel_size
            )
        elif downsample_type == "UNIFORM":
            target_points = self._settings.icp.downsample.target_uniform_point_count

            frame_point_cloud = frame.build_point_cloud(
                0.09, target_points=target_points)
        else:
            raise Exception(f"Unrecognized downsample type {downsample_type}")

        # First Iteration: No reference pose, we fix this as the origin of the coordinate system.
        if self._reference_point_cloud is None:
            frame._lidar_start_pose = self._reference_pose.clone(fixed=True, requires_tensor=True)
            frame._lidar_end_pose = self._reference_pose.clone(fixed=True, requires_tensor=True)
            self._reference_point_cloud = frame_point_cloud
            self._reference_time = (
                frame.start_image.timestamp + frame.end_image.timestamp) / 2
            self._velocity = torch.Tensor([0, 0, 0])
            self._angular_velocity = torch.Tensor([0, 0, 0])
            return True

        device = self._reference_pose.get_transformation_matrix().device

        # Future Iterations: Actually do ICP
        initial_guess = np.eye(4)

        # TODO: See the estimate normals section of http://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#
        # They recommend doing it a different way by setting a radius, but I don't know what radius to use
        self._reference_point_cloud.estimate_normals()
        frame_point_cloud.estimate_normals()

        convergence_criteria = (
            o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
                self._settings.icp.relative_fitness,
                self._settings.icp.relative_rmse,
                self._settings.icp.max_iterations,
            )
        )

        registration = o3d.pipelines.registration.registration_icp(
            frame_point_cloud,
            self._reference_point_cloud,
            self._settings.icp.threshold,
            initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=convergence_criteria,
        )

        registration_result = (
            torch.from_numpy(registration.transformation.copy()
                             ).float().to(device)
        )

        reference_pose_mat = self._reference_pose.get_transformation_matrix().detach()

        tracked_position = reference_pose_mat @ registration_result

        # Do interpolation/extrapolation to get the start/end lidar poses
        # TODO: Is there a better way to do this extrapolation?
        rot = R.from_matrix(registration_result[:3, :3])
        rot_vec = rot.as_rotvec()
        trans_vec = registration_result[:3, 3]

        new_reference_time = (
            frame.start_image.timestamp + frame.end_image.timestamp) / 2

        start_time_interp_factor = (
            frame.start_image.timestamp - self._reference_time) \
            / (new_reference_time - self._reference_time)
        rot_vec_start = rot_vec * start_time_interp_factor
        rot_start = torch.from_numpy(R.from_rotvec(rot_vec_start).as_matrix())
        trans_vec_start = start_time_interp_factor * trans_vec
        start_transformation_mat = torch.hstack(
            (rot_start, trans_vec_start.reshape(3, 1)))
        start_transformation_mat = torch.vstack(
            (start_transformation_mat, torch.Tensor([0, 0, 0, 1]))).float()
        frame._lidar_start_pose = Pose(
            (reference_pose_mat @ start_transformation_mat).float().to(device), requires_tensor=True)

        end_time_interp_factor = (frame.end_image.timestamp - self._reference_time) \
            / (new_reference_time - self._reference_time)

        rot_vec_end = rot_vec * end_time_interp_factor
        rot_end = torch.from_numpy(R.from_rotvec(rot_vec_end).as_matrix())
        trans_vec_end = end_time_interp_factor * trans_vec
        end_transformation_mat = torch.hstack(
            (rot_end, trans_vec_end.reshape(3, 1)))
        end_transformation_mat = torch.vstack(
            (end_transformation_mat, torch.Tensor([0, 0, 0, 1]))).float()
        frame._lidar_end_pose = Pose(
            (reference_pose_mat @ end_transformation_mat).float().to(device), requires_tensor=True)

        if self._settings.debug.write_icp_point_clouds:
            logdir = f"{self._settings.log_directory}/clouds/frame_{self._frame_count}"
            os.makedirs(logdir, exist_ok=True)
            o3d.io.write_point_cloud(
                f"{logdir}/reference_point_cloud.pcd", self._reference_point_cloud)
            o3d.io.write_point_cloud(
                f"{logdir}/frame_point_cloud.pcd", frame_point_cloud)
            o3d.io.write_point_cloud(
                f"{logdir}/transformed_frame_cloud.pcd",
                transform_cloud(frame_point_cloud, registration_result))
            np.savetxt(f"{logdir}/transform.txt", registration.transformation)

        self._reference_time = new_reference_time
        self._reference_pose = Pose(tracked_position, fixed=True)
        self._reference_point_cloud = frame_point_cloud

        self._frame_count += 1
        return True
