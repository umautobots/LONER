from common.frame import Frame


class KeyFrame:
    """ The KeyFrame class stores a frame an additional metadata to be used in optimziation.

    Note: Currently it's only a frame, but the class is kept separate as future-proofing.
    """

    ## Constructor: Create a KeyFrame from input Frame @p frame. 
    def __init__(self, frame: Frame) -> None:
        self._frame = frame

    def GetStartCameraPose(self) -> None:
        return self._frame.GetStartCameraPose()

    def GetEndCameraPose(self) -> None:
        return self._frame.GetEndCameraPose()

    def GetStartLidarPose(self) -> None:
        return self._frame.GetStartLidarPose()

    def GetEndLidarPose(self) -> None:
        return self._frame.GetEndLidarPose()


