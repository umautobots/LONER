import torch

from common.settings import Settings


class SkyRemoval:
    """ SkyRemoval implements segmentation to determine which pixels are sky.
    """

    # Constructor
    # @param settings: Settings object for sky removal
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # Given an image, return the binary mask where 1 is sky and 0 is not sky.
    def get_sky_mask(self, image: torch.Tensor) -> torch.Tensor:
        pass
