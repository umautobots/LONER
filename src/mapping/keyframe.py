from common.frame import Frame


class KeyFrame:
    """ The KeyFrame class stores a frame an additional metadata to be used in optimziation.

    Note: Currently it's only a frame, but the class is kept separate as future-proofing.
    """

    ## Constructor: Create a KeyFrame from input Frame @p frame. 
    def __init__(self, frame: Frame) -> None:
        self._frame = frame

    def get_start_camera_pose(self) -> None:
        return self._frame.get_start_camera_transform()

    def get_end_camera_pose(self) -> None:
        return self._frame.get_end_camera_transform()

    def get_start_lidar_pose(self) -> None:
        return self._frame.get_start_lidar_pose()

    def get_end_lidar_pose(self) -> None:
        return self._frame.get_end_lidar_pose()
