import torch

from common.pose_utils import WorldCube


class Image:
    """ Image class for holding images.

    A simple wrapper containing an image and a timestamp
    """

    # Constructor
    # @param image: a torch Tensor of RGB or Binary data
    # @param timestamp: the time at which the image was captured
    def __init__(self, image: torch.Tensor, timestamp: float):
        self.image = image
        self.timestamp = timestamp

    def to(self, device: int) -> None:
        self.image = self.image.to(device)
        
        if isinstance(self.timestamp, torch.Tensor):
            self.timestamp = self.timestamp.to(device)
        else:
            self.timestamp = torch.Tensor([self.timestamp]).to(device)

class LidarScan:
    """ LidarScan class for handling lidar data

    Represents Lidar data as ray directions, ray origin offsets, and timestamps.
    Note that this intentionally does not store the location of the pose. To reconstruct
    a point cloud, for each ray you would do the following, given a pose of lidar
    at time t of $$T_{l,t}$$:

    $$point = T_{l,timestamps[i]}*ray_origin_offsets[i] + ray_directions[i]$$
    """

    # Constructor
    # @param ray_directions: Direction of each ray. 3xn tensor.
    # @param distances: Distance of each ray. 1xn tensor
    # @param ray_origin_offsets: Offset from the origin of the lidar frame to the
    #        origin of each lidar ray. Either 4x4 tensor (constant offset) or 4x4xn tensor
    # @param timestamps: The time at which each laser fired. Used for motion compensation.
    def __init__(self,
                 ray_directions: torch.Tensor = torch.Tensor(),
                 distances: torch.Tensor = torch.Tensor(),
                 ray_origin_offsets: torch.Tensor = torch.eye(4),
                 timestamps: torch.Tensor = torch.Tensor()) -> None:

        self.ray_directions = ray_directions
        self.distances = distances
        self.ray_origin_offsets = ray_origin_offsets
        self.timestamps = timestamps

    def __len__(self) -> int:
        return self.timestamps.shape[0]

    def get_start_time(self):
        return self.timestamps[0]

    def get_end_time(self):
        return self.timestamps[-1]

    def clear(self):
        self.ray_directions = torch.Tensor()
        self.distances = torch.Tensor()
        self.ray_origin_offsets = torch.eye(4)
        self.timestamps = torch.Tensor()

    def remove_points(self, num_points):
        self.ray_directions = self.ray_directions[..., num_points:]
        self.distances = self.distances[num_points:]
        self.timestamps = self.timestamps[num_points:]

        single_origin = self.ray_origin_offsets.dim() == 2
        if not single_origin:
            self.ray_origin_offsets = self.ray_origin_offsets[..., num_points:]

    def merge(self, other: "LidarScan"):
        self.add_points(other.ray_directions,
                        other.distances,
                        other.ray_origin_offsets,
                        other.timestamps)

    def to(self, device):
        self.ray_directions = self.ray_directions.to(device)
        self.distances = self.distances.to(device)
        self.ray_origin_offsets = self.ray_origin_offsets.to(device)
        self.timestamps = self.timestamps.to(device)

    def add_points(self,
                   ray_directions: torch.Tensor,
                   distances: torch.Tensor,
                   ray_origin_offsets: torch.Tensor,
                   timestamps: torch.Tensor) -> None:

        if self.ray_directions.shape[0] == 0:
            self.distances = distances
            self.ray_directions = ray_directions
            self.ray_origin_offsets = ray_origin_offsets
            self.timestamps = timestamps
        else:
            self.ray_directions = torch.hstack(
                (self.ray_directions, ray_directions))
            self.timestamps = torch.hstack((self.timestamps, timestamps))
            self.distances = torch.hstack((self.distances, distances))

            # If it's a constant offset (4x4 tensor), then don't do anything
            if ray_origin_offsets.dim() == 3:
                self.ray_origin_offsets = torch.hstack(
                    (self.ray_origin_offsets, ray_origin_offsets))

    def transform_world_cube(self, world_cube: WorldCube, reverse=False) -> None:
        if reverse:
            self.ray_origin_offsets[...,:3,3] *= world_cube.scale_factor
            self.distances *= world_cube.scale_factor
        else:
            self.ray_origin_offsets /= world_cube.scale_factor
            self.ray_origin_offsets[...,:3,3] /= world_cube.scale_factor
