from typing import Union

import open3d as o3d
import torch

from common.sensors import Image, LidarScan
from src.common.pose import Pose

class Frame:
    """ Frame Class representing the atomic unit of optimization in Loner SLAM

    A Frame consists of an image, and lidar points that occured at nearby times.

    It also stores tensors used for computing the poses of cameras and lidars.
    """

    ## Constructor
    # @param image: The RGB Image
    # @param lidar_points: A LidarScan with points in the time range of the images
    # @param T_lidar_to_camera: Pose object with extrinsic calibration
    def __init__(self,
                 image: Image = None,
                 lidar_points: LidarScan = LidarScan(),
                 T_lidar_to_camera: Pose = None) -> None:

        self.image: Image = image
        self.lidar_points: LidarScan = lidar_points

        self._lidar_to_camera: Pose = T_lidar_to_camera

        self._lidar_pose: Pose = None
        self._gt_lidar_pose: Pose = None

        self._id = -1

    ## Creates a deepcopy of the frame. 
    # @returns a deepcopy of the current frame
    def clone(self) -> "Frame":
        
        attrs = ["image", "lidar_points", "_lidar_to_camera", "_lidar_pose", "_gt_lidar_pose"]

        new_frame = Frame()
        for attr in attrs:
            old_attr = getattr(self, attr)
            new_attr = None if old_attr is None else old_attr.clone()
            setattr(new_frame, attr, new_attr)

        return new_frame

    def __str__(self):
        im_str = "None" if self.image is None else self.image.timestamp
        if isinstance(im_str, torch.Tensor):
            im_str = im_str.item()

        return f"<Frame; Im: {im_str}, Pts: ({self.lidar_points.get_start_time()},{self.lidar_points.get_end_time()}))>"

    def __repr__(self):
        return self.__str__()

    ## Moves all items in the frame to the specified device, in-place. Also returns the current frame.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "Frame":
        if self.image is not None:
            self.image.to(device)
        
        self.lidar_points.to(device)

        for pose in [self._lidar_to_camera, self._lidar_pose, self._gt_lidar_pose]:
            if pose is not None:
                pose.to(device)

        return self

    ## Detaches the current frame from the computation graph
    # @returns a reference to self.
    def detach(self) -> "Frame":
        self._lidar_pose.detach()

        return self

    ## Gets the timestamp of the start of the scan
    def get_time(self) -> float:
        return self.lidar_points.get_start_time()

    def get_middle_time(self) -> float:
        return self.lidar_points.get_start_time()/2. + self.lidar_points.get_end_time()/2.

    ## Builds a point cloud from the lidar scan.
    # @p time_per_scan: The maximum time to allow in a scan. This prevents aliasing without motion compensation.
    # @p target_points: If not None, downsample uniformly to approximately this many points.
    # @returns a open3d Pointcloud
    def build_point_cloud(self, scan_duration: float = None,
                          target_points: int = None) -> o3d.cuda.pybind.geometry.PointCloud:
        pcd = o3d.cuda.pybind.geometry.PointCloud()

        if scan_duration is None:
            time_per_scan = None
        else:
            time_per_scan = scan_duration * self.get_scan_duration()

        # Only take 1 scan
        if time_per_scan is not None and self.lidar_points.get_end_time() - self.lidar_points.get_start_time() > 1e-3:
            lidar_timestamps = self.lidar_points.timestamps
            middle_time = (lidar_timestamps[0] + lidar_timestamps[-1])/2
            start_index = torch.argmax((lidar_timestamps - middle_time >= -time_per_scan/2).float())

            if lidar_timestamps[-1] < middle_time + time_per_scan/2:
                final_index = len(lidar_timestamps)
            else:
                final_index = torch.argmax((lidar_timestamps - middle_time >= time_per_scan/2).float())
        else:
            start_index = 0
            final_index = len(self.lidar_points.timestamps)

        if target_points is None:
            step_size = 1
        else:
            step_size = torch.div(
                final_index-start_index, target_points, rounding_mode='floor')

        if step_size == 0:
            step_size = 1
            
        end_points_local = self.lidar_points.ray_directions[..., start_index:final_index:step_size] * \
            self.lidar_points.distances[start_index:final_index:step_size]
        end_points_homog = torch.vstack(
            (end_points_local, torch.ones_like(end_points_local[0])))

        end_points_global = end_points_homog[:3, :]

        pcd.points = o3d.utility.Vector3dVector(
            end_points_global.cpu().numpy().transpose())
        return pcd
    
    def get_scan_duration(self) -> float:
        return (self.lidar_points.timestamps[-1] - self.lidar_points.timestamps[0]).item()

    ## @returns the Pose of the camera at the time the image was captured
    def get_camera_pose(self) -> Pose:
        return self._lidar_pose * self._lidar_to_camera

    ## @returns the Pose of the lidar at the time the image was captured
    def get_lidar_pose(self) -> Pose:
        return self._lidar_pose
