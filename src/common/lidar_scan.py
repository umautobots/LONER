import torch


class LidarScan:
    """ LidarScan class for handling lidar data
    
    Represents Lidar data as ray directions, ray origin offsets, and timestamps.
    Note that this intentionally does not store the location of the pose. To reconstruct
    a point cloud, for each ray you would do the following, given a pose of lidar
    at time t of $$T_{l,t}$$:

    $$point = T_{l,timestamps[i]}*ray_origin_offsets[i] + ray_directions[i]$$
    """
    
    ## Constructor
    # @param ray_directions: Direction of each ray. nx3 tensor.
    # @param ray_origin_offsets: Offset from the origin of the lidar frame to the 
    #        origin of each lidar ray. Either 4x4 tensor (constant offset) or 4x4xn tensor
    # @param timestamps: The time at which each laser fired. Used for motion compensation.
    def __init__(self,
                 ray_directions: torch.Tensor,
                 ray_origin_offsets: torch.Tensor,
                 timestamps: torch.Tensor) -> None:
        
        self._ray_directions = ray_directions
        self._ray_origin_offsets = ray_origin_offsets
        self._timestamps = timestamps