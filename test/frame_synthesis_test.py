from src.common.pose import Pose
from src.common.settings import Settings
from src.common.sensors import Image, LidarScan
from src.tracking.frame_synthesis import FrameSynthesis
import os
import sys
import torch
import unittest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


def BuildLidarScan(start_time, end_time, num_steps):
    timestamps = torch.linspace(start_time, end_time, num_steps)
    directions = torch.tile(timestamps, (3, 1))
    distances = torch.linalg.norm(directions, axis=0)
    directions = directions / distances
    offsets = torch.eye(4)
    return LidarScan(directions, distances, offsets, timestamps)


class TestFrameSynthesis(unittest.TestCase):
    def test_simple(self):

        with open("../cfg/default_settings.yaml") as yaml_file:
            settings_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        print(settings_dict)
        settings = Settings(settings_dict)
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1

        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(0, 0)
        image_1 = Image(0, 1)
        image_2 = Image(0, 2)
        image_3 = Image(0, 3)

        lidar_scan_01 = BuildLidarScan(0, 1.5, 10)
        lidar_scan_12 = BuildLidarScan(1.6, 2.5, 10)
        lidar_scan_23 = BuildLidarScan(2.6, 3.6, 10)

        fs.process_image(image_0)
        # print("===== Image 0 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_lidar(lidar_scan_01)
        # print("===== Lidar 01 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_image(image_1)
        # print("===== Image 1 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_lidar(lidar_scan_12)
        # print("===== Lidar 12 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_image(image_2)
        # print("===== Image 2 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_lidar(lidar_scan_23)
        # print("===== Lidar 23 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        fs.process_image(image_3)
        # print("===== Image 3 =====")
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # There should be two frames
        self.assertEqual(len(fs._completed_frames), 2)

        # The first frame should be images at t=[0,1]
        first_frame = fs._completed_frames[0]
        self.assertEqual(first_frame.start_image.timestamp, 0)
        self.assertEqual(first_frame.end_image.timestamp, 1)
        # lidar points should be in 0, 1.5
        dists = first_frame.lidar_points.timestamps - \
            torch.linspace(0, 1.5, 10)
        equal = torch.all(dists < 1e5)
        self.assertTrue(equal)

        # The second frame should be images at t=[2,3]
        second_frame = fs._completed_frames[1]
        self.assertEqual(second_frame.start_image.timestamp, 2)
        self.assertEqual(second_frame.end_image.timestamp, 3)
        # lidar points should be in 2.6, 3.5
        dists = second_frame.lidar_points.timestamps - \
            torch.linspace(1.6, 3.5, 19)
        equal = torch.all(dists < 1e5)
        self.assertTrue(equal)

    def test_small_lidar_scans(self):
        with open("../cfg/default_settings.yaml") as yaml_file:
            settings_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        print(settings_dict)
        settings = Settings(settings_dict)
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(0, 0)
        image_1 = Image(0, 1)
        image_2 = Image(0, 2)
        image_3 = Image(0, 3)

        lidar_scan_0 = BuildLidarScan(0, 0.5, 10)
        lidar_scan_1 = BuildLidarScan(0.6, 0.8, 10)
        lidar_scan_2 = BuildLidarScan(0.9, 1.5, 10)
        lidar_scan_3 = BuildLidarScan(1.6, 2.4, 10)
        lidar_scan_4 = BuildLidarScan(2.5, 2.9, 10)
        lidar_scan_5 = BuildLidarScan(3.0, 3.2, 10)
        lidar_scan_6 = BuildLidarScan(3.3, 3.6, 10)

        # print("===== Lidar 0 -> 0.5 =====")
        fs.process_lidar(lidar_scan_0)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Image 0 =====")
        fs.process_image(image_0)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 0.5->0.8 =====")
        fs.process_lidar(lidar_scan_1)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 0.9->1.5 =====")
        fs.process_lidar(lidar_scan_2)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Image 1 =====")
        fs.process_image(image_1)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Image 2 =====")
        fs.process_image(image_2)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 1.6->2.4 =====")
        fs.process_lidar(lidar_scan_3)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 2.5->2.9 =====")
        fs.process_lidar(lidar_scan_4)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 3.0->3.2 =====")
        fs.process_lidar(lidar_scan_5)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Image 3 =====")
        fs.process_image(image_3)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # print("===== Lidar 3.3->3.5 =====")
        fs.process_lidar(lidar_scan_6)
        # print("Active Frame:", fs._active_frame)
        # print("In Progress:", fs._in_progress_frames)
        # print("Completed:", fs._completed_frames)

        # There should be two frames
        self.assertEqual(len(fs._completed_frames), 2)

        # The first frame should be images at t=[0,1]
        first_frame = fs._completed_frames[0]
        self.assertEqual(first_frame.start_image.timestamp, 0)
        self.assertEqual(first_frame.end_image.timestamp, 1)
        self.assertAlmostEqual(first_frame.lidar_points.timestamps[0], 0)
        self.assertAlmostEqual(first_frame.lidar_points.timestamps[-1], 1.5)

        # The second frame should be images at t=[2,3]
        second_frame = fs._completed_frames[1]
        self.assertEqual(second_frame.start_image.timestamp, 2)
        self.assertEqual(second_frame.end_image.timestamp, 3)
        self.assertAlmostEqual(second_frame.lidar_points.timestamps[0], 1.6)
        self.assertAlmostEqual(second_frame.lidar_points.timestamps[-1], 3.5)

    def test_stale_lidar_points(self):
        with open("../cfg/default_settings.yaml") as yaml_file:
            settings_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        settings = Settings(settings_dict)
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(0, 0)
        image_1 = Image(0, 1)
        image_2 = Image(0, 2)
        image_3 = Image(0, 3)

        lidar_scan_01 = BuildLidarScan(0, 1.5, 10)
        lidar_scan_12 = BuildLidarScan(1.6, 2.5, 10)
        lidar_scan_23 = BuildLidarScan(2.6, 3.6, 10)

        fs.process_image(image_0)
        fs.process_image(image_1)
        fs.process_image(image_2)
        fs.process_lidar(lidar_scan_01)
        fs.process_lidar(lidar_scan_12)
        fs.process_lidar(lidar_scan_23)
        fs.process_image(image_3)

        # The first frame should be images at t=[0,1]
        first_frame = fs._completed_frames[0]
        self.assertEqual(first_frame.start_image.timestamp, 0)
        self.assertEqual(first_frame.end_image.timestamp, 1)
        self.assertAlmostEqual(first_frame.lidar_points.timestamps[0], 0)
        self.assertAlmostEqual(first_frame.lidar_points.timestamps[-1], 1.5)

        # The second frame should be images at t=[2,3]
        second_frame = fs._completed_frames[1]
        self.assertEqual(second_frame.start_image.timestamp, 2)
        self.assertEqual(second_frame.end_image.timestamp, 3)
        self.assertAlmostEqual(second_frame.lidar_points.timestamps[0], 1.6)
        expected_last_val = torch.linspace(2.6, 3.6, 10)[-2]
        self.assertAlmostEqual(
            second_frame.lidar_points.timestamps[-1].item(), expected_last_val)

    def test_big_lidar_scans(self):
        with open("../cfg/default_settings.yaml") as yaml_file:
            settings_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        settings = Settings(settings_dict)
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(0, 0)
        image_1 = Image(0, 1)
        image_2 = Image(0, 2)
        image_3 = Image(0, 3)
        image_4 = Image(0, 4)
        image_5 = Image(0, 5)

        lidar_scan_0 = BuildLidarScan(0, 4, 10)
        lidar_scan_1 = BuildLidarScan(4.1, 10, 10)

        fs.process_image(image_0)
        fs.process_image(image_1)
        fs.process_image(image_2)
        fs.process_image(image_3)
        fs.process_lidar(lidar_scan_0)
        fs.process_lidar(lidar_scan_1)
        fs.process_image(image_4)
        fs.process_image(image_5)

        # There should be two frames
        self.assertEqual(len(fs._completed_frames), 3)

        # The first frame should be images at t=[0,1]
        first_frame = fs._completed_frames[0]
        self.assertEqual(first_frame.start_image.timestamp, 0)
        self.assertEqual(first_frame.end_image.timestamp, 1)

        # The second frame should be images at t=[2,3]
        second_frame = fs._completed_frames[1]
        self.assertEqual(second_frame.start_image.timestamp, 2)
        self.assertEqual(second_frame.end_image.timestamp, 3)

        # The third frame should be images at t=[4,5]
        third_frame = fs._completed_frames[2]
        self.assertEqual(third_frame.start_image.timestamp, 4)
        self.assertEqual(third_frame.end_image.timestamp, 5)

        print(first_frame)
        print(second_frame)
        print(third_frame)


if __name__ == "__main__":
    unittest.main()
