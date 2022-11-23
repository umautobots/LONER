from typing import Dict

from common.frame import Frame
from common.settings import Settings
from mapping.keyframe import KeyFrame


class KeyFrameManager:
    """ The KeyFrame Manager class creates and manages KeyFrames and passes 
    data to the optimizer.
    """

    ## Constructor
    # settings: Settings object for the KeyFrame Manager
    def __init__(self, settings: Settings):
        self._settings = settings

    ## Processes the input @p frame, decides whether it's a KeyFrame, and if so
    ## adds it to internal KeyFrame storage
    def process_frame(self, frame: Frame):
        pass

    ## Selects which KeyFrames are to be used in the optimization, allocates 
    ## samples to them, and returns the result as {keyframe: num_samples}
    def get_active_window(self) -> Dict[KeyFrame, int]:
        pass