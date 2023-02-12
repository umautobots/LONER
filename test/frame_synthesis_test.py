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

from src.common.pose import Pose
from src.common.settings import Settings
from src.common.sensors import Image, LidarScan
from src.tracking.frame_synthesis import FrameSynthesis

def BuildLidarScan(start_time, end_time, num_steps):
    timestamps = torch.linspace(start_time, end_time, num_steps)
    directions = torch.tile(timestamps, (3, 1))
    distances = torch.linalg.norm(directions, axis=0)
    directions = directions / distances
    offsets = torch.eye(4)
    return LidarScan(directions, distances, offsets, timestamps)

"""
Old frame synthesis

    def try_synthesize_frame(self):
        assert self._use_simple_frames, "This only makes sense for simple frame synthesis"

        if len(self._images) == 0 or len(self._lidar_scans) == 0:
            return

        candidate_image = self._images[0]
        candidate_image_ts = candidate_image.timestamp

        # Haven't gotten a recent enough lidar scan yet
        if self._lidar_scans[-1].get_end_time() < candidate_image_ts - FRAME_TOLERANCE:
            return

        # Lidar scans are all after the image
        if self._lidar_scans[0].get_start_time() > candidate_image_ts + FRAME_TOLERANCE:
            # This image is doomed. We'll never get another lidar scan for it. Drop it
            self._images = self._images[1:]
            if len(self._images) > 0:
                self.try_synthesize_frame()
            else:
                return

        # We should be able to form a frame now
        chosen_scan_idx = torch.argmin(torch.abs(self._lidar_timestamps - candidate_image_ts)).item()

        assert torch.abs(self._lidar_timestamps[chosen_scan_idx] - candidate_image_ts) < self._frame_delta_t_sec + FRAME_TOLERANCE

        chosen_scan = self._lidar_scans[chosen_scan_idx]

        new_frame = SimpleFrame(candidate_image.clone(), chosen_scan, self._t_camera_to_lidar)

        self._images = self._images[1:]
        self._lidar_scans = self._lidar_scans[chosen_scan_idx+1:]
        self._lidar_timestamps = self._lidar_timestamps[chosen_scan_idx+1:]

        self._completed_frames.append(new_frame)

"""

class TestFrameSynthesis(unittest.TestCase):
    def test_simple(self):
        settings = Settings.load_from_file("../cfg/default_settings.yaml")
        fs_settings = settings.tracker.frame_synthesis

        fs_settings.use_simple_frames = False

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1


        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

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
        settings = Settings.load_from_file("../cfg/default_settings.yaml")

        fs_settings = settings.tracker.frame_synthesis
        fs_settings.use_simple_frames = False

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

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
        settings = Settings.load_from_file("../cfg/default_settings.yaml")

        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        fs_settings.use_simple_frames = False
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

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
        settings = Settings.load_from_file("../cfg/default_settings.yaml")
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.use_simple_frames = False

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)
        image_4 = Image(torch.tensor([0]), 4)
        image_5 = Image(torch.tensor([0]), 5)

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


class TestSimpleFrameSynthesis(unittest.TestCase):
    def test_simple(self):

        settings = Settings.load_from_file("../cfg/default_settings.yaml")
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        fs_settings.use_simple_frames = True
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

        lidar_scan_01 = BuildLidarScan(0, 1.5, 10)
        lidar_scan_12 = BuildLidarScan(1.6, 2.5, 10)
        lidar_scan_23 = BuildLidarScan(2.6, 3.6, 10)

        fs.process_image(image_0)
        fs.process_lidar(lidar_scan_01)
        fs.process_image(image_1)
        fs.process_lidar(lidar_scan_12)
        fs.process_image(image_2)
        fs.process_lidar(lidar_scan_23)
        fs.process_image(image_3) 

        self.assertEqual(len(fs._completed_frames), 4)
        self.assertEqual(fs._completed_frames[0].get_time(), 0)
        self.assertEqual(fs._completed_frames[1].get_time(), 1)
        self.assertEqual(fs._completed_frames[2].get_time(), 2)
        self.assertEqual(fs._completed_frames[3].get_time(), 3)


    def test_small_lidar_scans(self):
        settings = Settings.load_from_file("../cfg/default_settings.yaml")
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.use_simple_frames = True

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

        lidar_scan_0 = BuildLidarScan(0, 0.5, 10)
        lidar_scan_1 = BuildLidarScan(0.6, 0.8, 10)
        lidar_scan_2 = BuildLidarScan(0.9, 1.5, 10)
        lidar_scan_3 = BuildLidarScan(1.6, 2.4, 10)
        lidar_scan_4 = BuildLidarScan(2.5, 2.9, 10)
        lidar_scan_5 = BuildLidarScan(3.0, 3.2, 10)
        lidar_scan_6 = BuildLidarScan(3.3, 3.6, 10)

        fs.process_lidar(lidar_scan_0)
        fs.process_image(image_0)
        fs.process_lidar(lidar_scan_1)
        fs.process_lidar(lidar_scan_2)
        fs.process_image(image_1)
        fs.process_image(image_2)
        fs.process_lidar(lidar_scan_3)
        fs.process_lidar(lidar_scan_4)
        fs.process_lidar(lidar_scan_5)
        fs.process_image(image_3)
        fs.process_lidar(lidar_scan_6)

        self.assertEqual(len(fs._completed_frames), 4)

    def test_drop_image(self):
        settings = Settings.load_from_file("../cfg/default_settings.yaml")

        fs_settings = settings.tracker.frame_synthesis
        fs_settings.use_simple_frames = True

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)

        lidar_scan = BuildLidarScan(1, 10, 30)

        fs.process_image(image_0)
        fs.process_image(image_1)
        fs.process_image(image_2)
        fs.process_lidar(lidar_scan)
        fs.process_image(image_3)

        self.assertEqual(len(fs._completed_frames), 3)


    def test_big_lidar_scans(self):
        settings = Settings.load_from_file("../cfg/default_settings.yaml")
        
        fs_settings = settings.tracker.frame_synthesis
        fs_settings.use_simple_frames = True

        fs_settings.frame_decimation_rate_hz = 1
        fs_settings.lidar_point_step = 1
        extrinsics = Pose.from_settings(settings.calibration.lidar_to_camera)

        fs = FrameSynthesis(fs_settings, extrinsics)

        image_0 = Image(torch.tensor([0]), 0)
        image_1 = Image(torch.tensor([0]), 1)
        image_2 = Image(torch.tensor([0]), 2)
        image_3 = Image(torch.tensor([0]), 3)
        image_4 = Image(torch.tensor([0]), 4)
        image_5 = Image(torch.tensor([0]), 5)

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

        self.assertEqual(len(fs._completed_frames), 6)


if __name__ == "__main__":
    unittest.main()
