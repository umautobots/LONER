import torch

from common.sensors import Image, LidarScan
from common.pose_utils import Pose
import open3d as o3d

import cProfile
from profilehooks import profile

class Frame:
    """ Frame Class representing the atomic unit of optimization in ClonerSLAM
    
    A Frame F consists of a start image, end image, and lidar points that fell
    between the times at which those images were captured, our outside by an 
    amount <= delta_t (a constant).

    It also stores tensors used for computing the poses of cameras and lidars,
    and an (optional) mask classifying sky pixels.
    """

    ## Constructor
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
        
        self._lidar_to_camera: T_lidar_to_camera
        # TODO
        self._lidar_start_pose = None
        self._lidar_end_pose = None
        self._gt_lidar_start_pose = None
        self._gt_lidar_end_pose = None

    def __str__(self):
        start_im_str = "None" if self.start_image is None else self.start_image.timestamp
        end_im_str = "None" if self.end_image is None else self.end_image.timestamp
        return f"<Frame {start_im_str}->{end_im_str}, {self.lidar_points.timestamps})>"

    def __repr__(self):
        return self.__str__()

    ## Builds a point cloud from the lidar scan.
    # @p time_per_scan: The maximum time to allow in a scan. This prevents aliasing without motion compensation.
    # @p compensate_motion: If True, interpolate/extrapolate the lidar poses. If false, don't.
    # @p target_points: If not None, downsample uniformly to approximately this many points.
    # @returns 
    def BuildPointCloud(self, time_per_scan: float = None, 
                        compensate_motion: bool = False, 
                        target_points: int = None) -> o3d.cuda.pybind.geometry.PointCloud:
        pcd = o3d.cuda.pybind.geometry.PointCloud()

        # Only take 1 scan
        if time_per_scan is not None:
            final_index = torch.argmax((self.lidar_points.timestamps -\
                                        self.lidar_points.timestamps[0] >= time_per_scan).float())
        else:
            final_index = len(self.lidar_points.timestamps)

        if target_points is None:
            step_size = 1
        else:
            step_size = torch.div(final_index, target_points,rounding_mode='floor')

        if compensate_motion:
            raise NotImplementedError("Not yet implemented!")
        else:
            end_points_local = self.lidar_points.ray_directions[...,:final_index:step_size] * \
                               self.lidar_points.distances[:final_index:step_size]
            end_points_homog = torch.vstack((end_points_local, torch.ones_like(end_points_local[0])))

            if self.lidar_points.ray_origin_offsets.dim() == 3:
                raise NotImplementedError("Haven't added support for unique ray origins")
            
            end_points_global = (self.lidar_points.ray_origin_offsets @ end_points_homog)[:3, :]

        pcd.points = o3d.utility.Vector3dVector(end_points_global.cpu().numpy().transpose())
        return pcd

    def SetStartSkyMask(self, mask: Image) -> None:
        self.start_sky_mask = mask

    def SetEndSkyMask(self, mask: Image) -> None:
        self.end_sky_mask = mask

    ## Returns the Pose of the camera at the start of the frame as a transformation matrix
    def GetStartCameraTransform(self) -> torch.Tensor:
        return self._lidar_start_pose * self._lidar_to_camera

    ## Returns the Pose of the camera at the end of the frame as a transformation matrix
    def GetEndCameraTransform(self) -> torch.Tensor:
        return self._lidar_end_pose * self._lidar_to_camera

    ## Returns the Pose of the lidar at the start of the frame as a transformation matrix
    def GetStartLidarPose(self) -> Pose:
        return self._lidar_start_pose

    ## Returns the Pose of the lidar at the end of the frame as a transformation matrix
    def GetEndLidarPose(self) -> Pose:
        return self._lidar_end_pose


