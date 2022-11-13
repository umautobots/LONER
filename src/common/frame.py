import torch

from common.sensors import Image, LidarScan
from common.pose_utils import Pose

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

    def SetStartSkyMask(self, mask: Image) -> None:
        self.start_sky_mask = mask

    def SetEndSkyMask(self, mask: Image) -> None:
        self.end_sky_mask = mask

    ## Returns the Pose of the camera at the start of the frame as a transformation matrix
    def GetStartCameraPose(self) -> None:
        pass

    ## Returns the Pose of the camera at the end of the frame as a transformation matrix
    def GetEndCameraPose(self) -> None:
        pass

    ## Returns the Pose of the lidar at the start of the frame as a transformation matrix
    def GetStartLidarPose(self) -> None:
        pass

    ## Returns the Pose of the lidar at the end of the frame as a transformation matrix
    def GetEndLidarPose(self) -> None:
        pass


