from typing import Union

import numpy as np
import pytorch3d.transforms
import torch

from common.pose_utils import (WorldCube, tensor_to_transform,
                               transform_to_tensor)


class Pose:
    """ Class to define a possibly optimizable pose.
    Poses are represented as a 7-tuple [x,y,z,q_x,q_y,q_z,q_w]
    """

    ## Constructor
    # @param transformation_matrix: 4D Homogenous transformation matrix to turn into a pose
    # @param pose_tensor: An alternate argument to @p transformation matrix, allows you to
    #                     specify the 6-vector representation directly
    # @param fixed: Specifies whether or not a gradient should be computed.
    def __init__(self, transformation_matrix: torch.Tensor = torch.eye(4),
                 pose_tensor: torch.Tensor = None,
                 fixed: bool = None,
                 requires_tensor: bool = False):

        if fixed is None:
            if transformation_matrix is None:
                fixed = not pose_tensor.requires_grad
            else:
                fixed = not transformation_matrix.requires_grad
        
        if pose_tensor is not None:
            self._pose_tensor = pose_tensor
            self._pose_tensor.requires_grad_(not fixed)
            transformation_matrix = tensor_to_transform(self._pose_tensor).float()
        elif requires_tensor:
            # We do this copy back and forth to support computing gradients on the
            # resulting pose tensor. 
            self._pose_tensor = transform_to_tensor(transformation_matrix).float()
            self._pose_tensor.requires_grad_(not fixed)
            transformation_matrix = tensor_to_transform(self._pose_tensor).float()
        else:
            self._pose_tensor = None
            transformation_matrix = transformation_matrix.float()

        self._transformation_matrix = transformation_matrix
        self._transformation_matrix.requires_grad_(not fixed)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.get_transformation_matrix())

    ## Tells pytorch whether to compute a gradient (and hence consider optimizing) this pose.
    def set_fixed(self, fixed: bool = True) -> None:
        self._pose_tensor.requires_grad_(not fixed)

    ## Moves the pose in-place to the specified device, and @returns a reference to the current pose.
    def to(self, device: Union[str, int]) -> "Pose":
        if self._pose_tensor is not None:
            self._pose_tensor = self._pose_tensor.to(device)
        self._transformation_matrix = self._transformation_matrix.to(device)
        return self

    ## Returns a copy of the current pose that is detached from the computation graph.
    # @returns a new pose. 
    def detach(self) -> "Pose":
        return Pose(self.get_transformation_matrix().detach())

    ## Load in a setting dict of form {xyz: [x,y,z], "orientation": [w,x,y,z]} to a Pose
    # @returns a Pose representing the 
    def from_settings(pose_dict: dict, fixed: bool = True) -> "Pose":
        xyz = torch.Tensor(pose_dict['xyz'])
        quat = torch.Tensor(pose_dict['orientation'])

        axis_angle = pytorch3d.transforms.quaternion_to_axis_angle(quat)
        tensor = torch.cat((xyz, axis_angle))
        return Pose(pose_tensor=tensor, fixed=fixed)

    ## Converts the current Pose to a dict and @returns the pose as a dict.
    def to_settings(self) -> dict:
        translation = self.get_translation().detach().cpu()
        xyz = [translation[i].item() for i in range(3)]

        quat = pytorch3d.transforms.matrix_to_quaternion(self.get_rotation().detach().cpu())

        return {
            "xyz": xyz,
            "orientation": quat
        }

    ## Transforms the pose according to the transformation represented by @p world_cube.
    # @param reverse specifies whether to invert the transformation
    # @param ignore_shift: If set, scale only and don't shift.
    # @returns self
    def transform_world_cube(self, world_cube: WorldCube, reverse=False, ignore_shift=False) -> "Pose":
        if reverse:
            self.get_translation()[:3,3] *= world_cube.scale_factor
            if not ignore_shift:
                self.get_translation()[:3,3] -= world_cube.shift
        else:
            if not ignore_shift:
                self.get_translation()[:3,3] += world_cube.shift
            self.get_translation()[:3,3] /= world_cube.scale_factor

        return self

    ## @returns a copy of the current pose.
    def clone(self, fixed=None, requires_tensor=False) -> "Pose":
        if fixed is None:
            fixed = not self.get_transformation_matrix().requires_grad
        return Pose(self.get_transformation_matrix().clone(), fixed=fixed, requires_tensor=requires_tensor)

    # Performs matrix multiplication on matrix representations of the given poses, and returns the result
    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.get_transformation_matrix() @ other.get_transformation_matrix())

    # Inverts the transformation represented by the pose
    def inv(self) -> "Pose":
        inv_mat = self.get_transformation_matrix().inverse()
        new_pose = Pose(inv_mat)

        return new_pose

    ## Gets the matrix representation of the pose. Only pytorch operations are used, so gradients are preserved.
    # @returns a 4x4 homogenous transformation matrix
    def get_transformation_matrix(self) -> torch.Tensor:
        if self._pose_tensor is None or not self._pose_tensor.requires_grad:
            return self._transformation_matrix

        return tensor_to_transform(self._pose_tensor)

    ## Gets the underlying 7-tensor. Should basically never be used.
    def get_pose_tensor(self) -> torch.Tensor:
        if self._pose_tensor is None:
            self._pose_tensor = transform_to_tensor(self.get_transformation_matrix())
        return self._pose_tensor

    ## Gets the translation component of the pose
    def get_translation(self) -> torch.Tensor:
        if self._pose_tensor is not None:
            return self._pose_tensor[:3]
        return self.get_transformation_matrix()[:3, 3]

    ## Returns the rotation as a rotation matrix
    def get_rotation(self) -> torch.Tensor:
        return self.get_transformation_matrix()[:3,:3]

    ## Converts the rotation to an axis-angle representation for interpolation
    def get_axis_angle(self) -> torch.Tensor:
        if self._pose_tensor is not None:
            return self._pose_tensor[3:]
        pytorch3d.transforms.matrix_to_axis_angle(self.get_rotation())
