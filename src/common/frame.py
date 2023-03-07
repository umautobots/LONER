from typing import Union

import open3d as o3d
import torch

from common.sensors import Image, LidarScan
from src.common.pose import Pose


class Frame:
    """ Frame Class representing the atomic unit of optimization in ClonerSLAM

    A Frame F consists of a start image, end image, and lidar points that fell
    between the times at which those images were captured, our outside by an 
    amount <= delta_t (a constant).

    It also stores tensors used for computing the poses of cameras and lidars,
    and an (optional) mask classifying sky pixels.
    """

    # Constructor
    # @param start_image: The RGB Image at the start of the frame
    # @param end_image: The RGB Image at the end of the frame
    # @param lidar_points: A LidarScan with points in the time range of the images (plus delta_t)
    # @param T_lidar_to_camera: Pose object with extrinsic calibration
    # @param start_sky_mask [optional]: Binary mask where 1 indicates sky and 0 is not sky in first image.
    # @param end_sky_mask [optional]: Binary mask where 1 indicates sky and 0 is not sky in second image.
    def __init__(self,
                 start_image: Image = None,
                 end_image: Image = None,
                 lidar_points: LidarScan = LidarScan(),
                 T_lidar_to_camera: Pose = None,
                 start_sky_mask: Image = None,
                 end_sky_mask: Image = None) -> None:

        self.start_image = start_image
        self.end_image = end_image
        self.lidar_points = lidar_points
        self.start_sky_mask = start_sky_mask
        self.end_sky_mask = end_sky_mask

        self._lidar_to_camera = T_lidar_to_camera

        self._lidar_start_pose: Pose = None
        self._lidar_end_pose: Pose = None
        self._gt_lidar_start_pose: Pose = None
        self._gt_lidar_end_pose: Pose = None

    ## Returns a deep-copy of the current frame
    def clone(self) -> "Frame":
        
        attrs = ["start_image", "end_image", "lidar_points", "_lidar_to_camera", "start_sky_mask",
                 "end_sky_mask", "_lidar_start_pose", "_lidar_end_pose", "_gt_lidar_start_pose",
                 "_gt_lidar_end_pose"]

        new_frame = Frame()
        for attr in attrs:
            old_attr = getattr(self, attr)
            new_attr = None if old_attr is None else old_attr.clone()
            setattr(new_frame, attr, new_attr)

        return new_frame

    def __str__(self):
        start_im_str = "None" if self.start_image is None else self.start_image.timestamp
        end_im_str = "None" if self.end_image is None else self.end_image.timestamp
        if isinstance(start_im_str, torch.Tensor):
            start_im_str = start_im_str.item()
        if isinstance(end_im_str, torch.Tensor):
            end_im_str = end_im_str.item()

        return f"<Frame {start_im_str}->{end_im_str}, {len(self.lidar_points.timestamps)} points)>"

    def __repr__(self):
        return self.__str__()

    ## Moves all items in the frame to the specified device, in-place. Also returns the current frame.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "Frame":
        self.start_image.to(device)
        self.end_image.to(device)
        self.lidar_points.to(device)

        for pose in [self._lidar_to_camera, self._lidar_start_pose, self._lidar_end_pose,
                     self._gt_lidar_start_pose, self._gt_lidar_end_pose]:
            if pose is not None:
                pose.to(device)

        return self

    ## Detaches the current frame from the computation graph, and @returns a reference to self.
    def detach(self) -> "Frame":
        self._lidar_start_pose.detach()
        self._lidar_end_pose.detach()

        return self

    def get_start_time(self):
        return self.start_image.timestamp

    def get_end_time(self):
        return self.end_image.timestamp

    ## Builds a point cloud from the lidar scan.
    # @p time_per_scan: The maximum time to allow in a scan. This prevents aliasing without motion compensation.
    # @p compensate_motion: If True, interpolate/extrapolate the lidar poses. If false, don't.
    # @p target_points: If not None, downsample uniformly to approximately this many points.
    # @returns a open3d Pointcloud
    def build_point_cloud(self, time_per_scan: float = None,
                          compensate_motion: bool = False,
                          target_points: int = None) -> o3d.cuda.pybind.geometry.PointCloud:
        pcd = o3d.cuda.pybind.geometry.PointCloud()

        # Only take 1 scan
        if time_per_scan is not None:
            final_index = torch.argmax((self.lidar_points.timestamps -
                                        self.lidar_points.timestamps[0] >= time_per_scan).float())
        else:
            final_index = len(self.lidar_points.timestamps)

        if target_points is None:
            step_size = 1
        else:
            step_size = torch.div(
                final_index, target_points, rounding_mode='floor')

        if compensate_motion:
            raise NotImplementedError("Not yet implemented!")
        else:
            end_points_local = self.lidar_points.ray_directions[..., :final_index:step_size] * \
                self.lidar_points.distances[:final_index:step_size]
            end_points_homog = torch.vstack(
                (end_points_local, torch.ones_like(end_points_local[0])))

            end_points_global = end_points_homog[:3, :]

        pcd.points = o3d.utility.Vector3dVector(
            end_points_global.cpu().numpy().transpose())
        return pcd

    def set_start_sky_mask(self, mask: Image) -> "Frame":
        self.start_sky_mask = mask
        return self

    def set_end_sky_mask(self, mask: Image) -> "Frame":
        self.end_sky_mask = mask
        return self

    ## @returns the Pose of the camera at the start of the frame as a transformation matrix
    def get_start_camera_pose(self) -> Pose:
        return self._lidar_start_pose * self._lidar_to_camera

    ## @returns the Pose of the camera at the end of the frame as a transformation matrix
    def get_end_camera_pose(self) -> Pose:
        return self._lidar_end_pose * self._lidar_to_camera

    ## @returns the Pose of the lidar at the start of the frame as a transformation matrix
    def get_start_lidar_pose(self) -> Pose:
        return self._lidar_start_pose

    ## @returns the Pose of the lidar at the end of the frame as a transformation matrix
    def get_end_lidar_pose(self) -> Pose:
        return self._lidar_end_pose

class SimpleFrame:
    """ Frame Class representing the atomic unit of optimization in ClonerSLAM

    A Frame F consists of a start image, end image, and lidar points that fell
    between the times at which those images were captured, our outside by an 
    amount <= delta_t (a constant).

    It also stores tensors used for computing the poses of cameras and lidars,
    and an (optional) mask classifying sky pixels.
    """

    # Constructor
    # @param start_image: The RGB Image at the start of the frame
    # @param end_image: The RGB Image at the end of the frame
    # @param lidar_points: A LidarScan with points in the time range of the images (plus delta_t)
    # @param T_lidar_to_camera: Pose object with extrinsic calibration
    # @param start_sky_mask [optional]: Binary mask where 1 indicates sky and 0 is not sky in first image.
    # @param end_sky_mask [optional]: Binary mask where 1 indicates sky and 0 is not sky in second image.
    def __init__(self,
                 image: Image = None,
                 lidar_points: LidarScan = LidarScan(),
                 T_lidar_to_camera: Pose = None) -> None:

        self.image: Image = image
        self.lidar_points: LidarScan = lidar_points

        self._lidar_to_camera: Pose = T_lidar_to_camera

        self._lidar_pose: Pose = None
        self._gt_lidar_pose: Pose = None

    ## Returns a deep-copy of the current frame
    def clone(self) -> "SimpleFrame":
        
        attrs = ["image", "lidar_points", "_lidar_to_camera", "_lidar_pose", "_gt_lidar_pose"]

        new_frame = SimpleFrame()
        for attr in attrs:
            old_attr = getattr(self, attr)
            new_attr = None if old_attr is None else old_attr.clone()
            setattr(new_frame, attr, new_attr)

        return new_frame

    def __str__(self):
        im_str = "None" if self.image is None else self.image.timestamp
        if isinstance(im_str, torch.Tensor):
            im_str = im_str.item()

        return f"<SimpleFrame; Im: {im_str}, Pts: ({self.lidar_points.get_start_time()},{self.lidar_points.get_end_time()}))>"

    def __repr__(self):
        return self.__str__()

    ## Moves all items in the frame to the specified device, in-place. Also returns the current frame.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "SimpleFrame":
        self.image.to(device)
        self.lidar_points.to(device)

        for pose in [self._lidar_to_camera, self._lidar_pose, self._gt_lidar_pose]:
            if pose is not None:
                pose.to(device)

        return self

    ## Detaches the current frame from the computation graph, and @returns a reference to self.
    def detach(self) -> "SimpleFrame":
        self._lidar_pose.detach()

        return self

    def get_time(self) -> float:
        return self.image.timestamp

    ## Builds a point cloud from the lidar scan.
    # @p time_per_scan: The maximum time to allow in a scan. This prevents aliasing without motion compensation.
    # @p target_points: If not None, downsample uniformly to approximately this many points.
    # @returns a open3d Pointcloud
    def build_point_cloud(self, time_per_scan: float = None,
                          target_points: int = None) -> o3d.cuda.pybind.geometry.PointCloud:
        pcd = o3d.cuda.pybind.geometry.PointCloud()

        # Only take 1 scan
        if time_per_scan is not None:
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
            breakpoint()
            step_size = 1
            
        end_points_local = self.lidar_points.ray_directions[..., start_index:final_index:step_size] * \
            self.lidar_points.distances[start_index:final_index:step_size]
        end_points_homog = torch.vstack(
            (end_points_local, torch.ones_like(end_points_local[0])))

        end_points_global = end_points_homog[:3, :]

        pcd.points = o3d.utility.Vector3dVector(
            end_points_global.cpu().numpy().transpose())
        return pcd

    ## @returns the Pose of the camera at the time the image was captured
    def get_camera_pose(self) -> Pose:
        return self._lidar_pose * self._lidar_to_camera

    ## @returns the Pose of the lidar at the time the image was captured
    def get_lidar_pose(self) -> Pose:
        return self._lidar_pose
