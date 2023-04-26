import cProfile
import copy
import os
import time

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp

from common.frame import Frame
from common.pose import Pose
from common.settings import Settings
from common.signals import Signal, StopSignal
from tracking.frame_synthesis import FrameSynthesis
import kornia.morphology

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
    # @param settings: Top level settings for the entire Loner SLAM module. Needed for calib etc.
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

        self._lidar_only = settings.system.lidar_only

        self._frame_synthesizer = FrameSynthesis(
            self._settings.frame_synthesis, self._t_lidar_to_camera, self._lidar_only)

        # Used to indicate to an external process that I've processed the stop signal
        self._processed_stop_signal = mp.Value("i", 0)
        # Set to 0 from an external thread when it's time to actuall exit.
        self._term_signal = mp.Value("i", 0)

        # Used for frame-to-frame ICP tracking
        self._reference_point_cloud = None
        self._reference_pose = Pose(fixed=True)
        self._reference_time = None

        self._frame_count = 0
        self._last_mapped_frame_time = None
        self._last_tracked_frame_time = 0

        self._frame_rate = self._settings.frame_synthesis.frame_decimation_rate_hz
        self._max_time_delta = self._settings.synchronization.max_time_delta


    def update(self):
        
        tic = time.time()
        num_tracked = 0

        if self._settings.synchronization.enabled and self._last_mapped_frame_time is not None:
            while self._last_tracked_frame_time > self._last_mapped_frame_time.value + self._max_time_delta:
                time.sleep(0.01)

        if self._processed_stop_signal.value:
            print("Not updating tracker: Tracker already done.")

        if self._rgb_slot.has_value():
            new_rgb = self._rgb_slot.get_value()

            if isinstance(new_rgb, StopSignal):
                self._processed_stop_signal.value = 1
                return

            self._frame_synthesizer.process_image(new_rgb)

        if self._lidar_slot.has_value():
            val = self._lidar_slot.get_value()

            if isinstance(val, StopSignal):
                self._processed_stop_signal.value = 1
                return

            new_lidar, new_gt_pose = val

            self._frame_synthesizer.process_lidar(new_lidar, new_gt_pose)

        while self._frame_synthesizer.has_frame():
            frame = self._frame_synthesizer.pop_frame()
            frame._id = self._frame_count
            tracked = self.track_frame(frame)

            if not tracked:
                print("Warning: Failed to track frame. Skipping.")
                continue

            if self._settings.compute_sky_rays:
                self.compute_sky_rays(frame)
            
            if self._settings.debug.write_frame_point_clouds:
                pcd = frame.build_point_cloud()
                logdir = f"{self._settings.log_directory}/frames/"
                os.makedirs(logdir, exist_ok=True)
                o3d.io.write_point_cloud(
                    f"{logdir}/cloud_{self._frame_count}.pcd", pcd)

                if frame.lidar_points.sky_rays is not None:
                    dummy_frame = Frame(None, frame.lidar_points.get_sky_scan(100))
                    pcd = dummy_frame.build_point_cloud()
                    o3d.io.write_point_cloud(
                        f"{logdir}/cloud_{self._frame_count}_sky.pcd", pcd)

            self._frame_signal.emit(frame)
            self._frame_count += 1
            self._last_tracked_frame_time = frame.get_time()
            num_tracked += 1

        toc = time.time()
        
        if num_tracked > 0 and self._settings.debug.log_times:
            with open(f"{self._settings.log_directory}/track_times.csv", "a+") as time_f:
                time_f.write(f"{toc - tic},{num_tracked}\n")

    ## Run spins and processes incoming data while putting resulting frames into the queue
    def run(self, last_mapped_frame_time) -> None:
        self._last_mapped_frame_time = last_mapped_frame_time
        while not self._processed_stop_signal.value:
            self.update()

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
            frame_point_cloud = frame.build_point_cloud(scan_duration=self._settings.icp.scan_duration)
        elif downsample_type == "VOXEL":
            frame_point_cloud = frame.build_point_cloud(scan_duration=self._settings.icp.scan_duration)
            voxel_size = self._settings.icp.downsample.voxel_downsample_size
            frame_point_cloud = frame_point_cloud.voxel_down_sample(
                voxel_size=voxel_size
            )
        elif downsample_type == "UNIFORM":
            target_points = self._settings.icp.downsample.target_uniform_point_count

            frame_point_cloud = frame.build_point_cloud(scan_duration=self._settings.icp.scan_duration,target_points=target_points)
        else:
            raise Exception(f"Unrecognized downsample type {downsample_type}")

        # First Iteration: No reference pose, we fix this as the origin of the coordinate system.
        if self._reference_point_cloud is None:
            frame._lidar_pose = self._reference_pose.clone(fixed=True, requires_tensor=True)
            self._reference_point_cloud = frame_point_cloud
            self._reference_point_cloud.estimate_normals()

            self._reference_time = frame.get_middle_time()

            self._velocity = torch.Tensor([0, 0, 0])
            self._angular_velocity = torch.Tensor([0, 0, 0])
            return True

        device = self._reference_pose.get_transformation_matrix().device

        # Future Iterations: Actually do ICP
        initial_guess = np.eye(4)

        frame_point_cloud.estimate_normals()

        for icp_settings in self._settings.icp.schedule:
            convergence_criteria = (
                o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(
                    icp_settings.relative_fitness,
                    icp_settings.relative_rmse,
                    icp_settings.max_iterations,
                )
            )
            registration = o3d.pipelines.registration.registration_icp(
                frame_point_cloud,
                self._reference_point_cloud,
                icp_settings.threshold,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=convergence_criteria,
            )
            initial_guess = registration.transformation.copy()

        registration_result = torch.from_numpy(initial_guess).float().to(device)

        reference_pose_mat = self._reference_pose.get_transformation_matrix().detach()

        tracked_position = reference_pose_mat @ registration_result

        new_reference_time = frame.get_middle_time()
        frame._lidar_pose = Pose(tracked_position.float().to(device), requires_tensor=True)
        
        if self._settings.motion_compensation.enabled:
            mocomp_poses = (self._reference_pose, frame._lidar_pose)
            mocomp_times = (self._reference_time, new_reference_time)
            use_gpu = self._settings.motion_compensation.use_gpu

            frame.lidar_points.motion_compensate(mocomp_poses, mocomp_times, frame._lidar_pose, use_gpu)
                
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
        return True

    def compute_sky_rays(self, frame: Frame):
        
        TOP_ROWS = 3
        HORIZON_OFFSET=10

        dirs = frame.lidar_points.ray_directions
        x,y,z = dirs[0], dirs[1], dirs[2]
        theta = torch.atan2(y, x).rad2deg().round().long()
        phi = torch.atan2(torch.sqrt(x**2 + y**2), z).rad2deg().round().long()

        phi_img = phi - phi.min()
        theta_img = theta-theta.min()
        theta_img[theta_img == 360] = 0

        depth_img = torch.zeros((phi_img.max()+1, 360), device=phi.device)
            
        depth_img[phi_img, theta_img] = 1

        depth_img = kornia.morphology.dilation(depth_img.unsqueeze(0).unsqueeze(0), torch.ones((3,3), device=depth_img.device))
        depth_img = kornia.morphology.erosion(depth_img, torch.ones((3,3), device=depth_img.device)).squeeze(0).squeeze(0)

        depth_img[:TOP_ROWS] = 1

        zero_locs = torch.where(depth_img == 0)
        zero_phi = (zero_locs[0] + phi.min()).deg2rad()
        zero_theta = (zero_locs[1] + theta.min()).deg2rad()

        z_out = torch.cos(zero_phi)
        y_out = torch.sin(zero_phi) * torch.sin(zero_theta)
        x_out = torch.sin(zero_phi) * torch.cos(zero_theta)

        zero_dirs = torch.vstack((x_out, y_out, z_out))

        r = frame.get_lidar_pose().get_rotation()

        zero_dirs_world = r @ zero_dirs
        x_w,y_w,z_w = zero_dirs_world[0], zero_dirs_world[1], zero_dirs_world[2]
        phi_w = 90 - torch.atan2(torch.sqrt(x_w**2 + y_w**2), z_w).rad2deg()
        sky_rays = zero_dirs_world[:, phi_w > HORIZON_OFFSET]

        frame.lidar_points.sky_rays = sky_rays