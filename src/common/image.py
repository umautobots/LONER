import torch


class Image:
    """ Image class for holding images.

    A simple wrapper containing an image and a timestamp
    """

    ## Constructor
    # @param image: a torch Tensor of RGB or Binary data
    # @param timestamp: the time at which the image was captured
    def __init__(self, image: torch.Tensor, timestamp: torch.Tensor):
        self.image = image
        self.timestamp = timestamp