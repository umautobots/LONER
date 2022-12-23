import torch
from typing import Union
import numpy as np
import pytorch3d.transforms


from common.pose_utils import transform_to_tensor, tensor_to_transform, WorldCube


class Pose:
    """ Class to define an optimizable pose.
    Poses are represented as a 7-tuple [x,y,z,q_x,q_y,q_z,q_w]
    """

    # Constructor
    # @param transformation_matrix: 4D Homogenous transformation matrix to turn into a pose
    # @param pose_tensor: An alternate argument to @p transformation matrix, allows you to
    #                 specify the 7-tuple representation directly
    # @param fixed: Specifies whether or not a gradient should be computed.
    def __init__(self, transformation_matrix: torch.Tensor = torch.eye(4),
                 pose_tensor: torch.Tensor = None,
                 fixed: bool = False):

        if pose_tensor is not None:
            self._pose_tensor = pose_tensor
        else:
            self._pose_tensor = transform_to_tensor(transformation_matrix)

    # Tells pytorch whether to compute a gradient (and hence consider optimizing) this pose.
    def set_fixed(self, fixed: bool = True) -> None:
        self._pose_tensor.requires_grad_(not fixed)

    # Moves the pose in-place to the specified device, and returns self.
    def to(self, device: Union[str, int]) -> "Pose":
        self._pose_tensor = self._pose_tensor.to(device)
        return self

    def detach(self) -> "Pose":
        self._pose_tensor = self._pose_tensor.detach()

    # Load in a setting dict of form {xyz: [x,y,z], "orientation": [x,y,z,w]} to a Pose
    def from_settings(pose_dict: dict, fixed: bool = False) -> "Pose":
        xyz = torch.Tensor(pose_dict['xyz'])
        quat = torch.Tensor(pose_dict['orientation'])

        tensor = torch.cat((xyz, quat))
        return Pose(pose_tensor=tensor, fixed=fixed)

    # Transforms the pose according to the transformation represented by @p world_cube.
    # @param reverse specifies whether to invert the transformation
    # @param ignore_shift: If set, scale only and don't shift.
    # @returns self
    def transform_world_cube(self, world_cube: WorldCube, reverse=False, ignore_shift=False) -> "Pose":
        if reverse:
            self._pose_tensor[:3] *= world_cube.scale_factor
            if not ignore_shift:
                self._pose_tensor[:3] -= world_cube.shift
        else:
            self._pose_tensor[:3] += world_cube.shift
            if not ignore_shift:
                self._pose_tensor[:3] /= world_cube.scale_factor

        return self

    # @returns a copy of the current pose.
    def clone(self, fixed=None) -> "Pose":
        if fixed is None:
            fixed = self._pose_tensor.requires_grad
        return Pose(pose_tensor=self._pose_tensor.clone(), fixed=fixed)

    # Performs matrix multiplication on matrix representations of the given poses, and returns the result
    def __mul__(self, other) -> "Pose":
        return Pose(self.get_transformation_matrix() @ other.get_transformation_matrix())

    # Inverts the transformation represented by the pose
    def inv(self) -> "Pose":
        return Pose(torch.linalg.inv(self.get_transformation_matrix()))

    # Gets the matrix representation of the pose. Only pytorch operations are used, so gradients are preserved.
    def get_transformation_matrix(self) -> torch.Tensor:
        return tensor_to_transform(self._pose_tensor)

    # Gets the underlying 7-tensor
    def get_pose_tensor(self) -> torch.Tensor:
        return self._pose_tensor

    def get_translation(self) -> torch.Tensor:
        return self._pose_tensor[:3]

    # Returns the rotation.
    # @param real_first: If set, returns a quaternion [w,x,y,z] instead of [x,y,z,w] (which is the default)
    def get_rotation(self, real_first=False) -> torch.Tensor:
        if real_first:
            return torch.cat([self._pose_tensor[-1], self._pose_tensor[3:6]])
        return self._pose_tensor[3:]

    # Converts the rotation to an axis-angle representation for interpolation
    def get_axis_angle(self) -> torch.Tensor:
        pytorch3d.transforms.quaternion_to_axis_angle(self.get_rotation(True))
