import yaml
import numpy as np
import os


class FusionPortableCalibration:
    """
    Input: Path to the top-level calibration directory (i.e. 20220209_calib)   
    """

    def __init__(self, calibration_path: str) -> None:

        class CvMatLoader(yaml.SafeLoader):
            def __init__(self, stream):
                self._root = os.path.split(stream.name)[0]
                super().__init__(stream)

            def cv_matrix(self, node):
                mat_data = self.construct_mapping(node, True)
                num_rows = mat_data["rows"]
                num_cols = mat_data["cols"]

                dt = mat_data["dt"]

                assert dt == "f", "Only floats supported. Fix if needed"

                data = mat_data["data"]

                mat = np.array(data).astype(float)

                if num_rows > 1:
                    mat = mat.reshape(num_rows, num_cols)

                return mat

        # Bit hacky due to super weird format opencv dumped the data to.
        # Using OpenCV file storage would require adding yaml version tags, which causes other problems
        CvMatLoader.add_constructor(
            u"tag:yaml.org,2002:opencv-matrix", CvMatLoader.cv_matrix)

        with open(f"{calibration_path}/calib/ouster00.yaml") as lidar_cal_file:
            lidar_cal = yaml.load(lidar_cal_file, Loader=CvMatLoader)
            frame_0_quat = lidar_cal["quaternion_sensor_frame_cam00"]
            frame_0_quat[0], frame_0_quat[3] = frame_0_quat[3], frame_0_quat[0]
            frame_0_trans = lidar_cal["translation_sensor_frame_cam00"]

            self.t_lidar_to_left_cam = {"translation": frame_0_trans,
                                        "rotation": frame_0_quat}

        with open(f"{calibration_path}/calib/frame_cam00.yaml") as frame0_cal_file:
            frame0_cal = yaml.load(frame0_cal_file, Loader=CvMatLoader)
            K = frame0_cal["camera_matrix"]
            distortion_coeffs = frame0_cal["distortion_coefficients"]
            distortion_model = frame0_cal["distortion_model"]
            rectification_matrix = frame0_cal["rectification_matrix"]
            projection_matrix = frame0_cal["projection_matrix"]

            self.left_cam_intrinsic = {
                "K": K,
                "distortion_model": distortion_model,
                "distortion_coeffs": distortion_coeffs,
                "rectification_matrix": rectification_matrix,
                "projection_matrix": projection_matrix,
                "width": frame0_cal["image_width"],
                "height": frame0_cal["image_height"]
            }

        with open(f"{calibration_path}/calib/frame_cam01.yaml") as frame1_cal_file:
            frame1_cal = yaml.load(frame1_cal_file, Loader=CvMatLoader)

            K = frame1_cal["camera_matrix"]
            distortion_coeffs = frame1_cal["distortion_coefficients"]
            distortion_model = frame1_cal["distortion_model"]
            rectification_matrix = frame1_cal["rectification_matrix"]
            projection_matrix = frame1_cal["projection_matrix"]

            self.right_cam_intrinsic = {
                "K": K,
                "distortion_model": distortion_model,
                "distortion_coeffs": distortion_coeffs,
                "rectification_matrix": rectification_matrix,
                "projection_matrix": projection_matrix,
                "width": frame1_cal["image_width"],
                "height": frame1_cal["image_height"]
            }

            stereo_rotation = frame1_cal["quaternion_stereo"]
            stereo_rotation[0], stereo_rotation[3] = stereo_rotation[3], stereo_rotation[0]
            frame1_cal["quaternion_stereo"]
            self.t_left_cam_to_right_cam = {
                "translation": frame1_cal["translation_stereo"],
                "rotation": stereo_rotation
            }
            self.stereo_baseline = np.linalg.norm(
                frame1_cal["translation_stereo"])
