import torch
import scipy.spatial.transform as sptransform
import numpy as np
import pytorch3d.transforms


from common.pose_utils import transform_to_tensor, tensor_to_transform, WorldCube


class Pose:
    def __init__(self, transformation_matrix: torch.Tensor = torch.eye(4),
                 pose_tensor: torch.Tensor = None,
                 fixed: bool = False):

        if pose_tensor is not None:
            self._pose_tensor = pose_tensor
        else:
            self._pose_tensor = transform_to_tensor(transformation_matrix)
        self._pose_tensor.requires_grad_(not fixed)

    def from_settings(pose_dict: dict, fixed: bool = False) -> "Pose":
        xyz = torch.Tensor(pose_dict['xyz'])
        quat = torch.Tensor(pose_dict['orientation'])

        tensor = torch.cat((xyz, quat))
        return Pose(pose_tensor=tensor, fixed=fixed)

    def transform_world_cube(self, world_cube: WorldCube, reverse=False) -> "Pose":
        if reverse:
            self._pose_tensor[:3] *= world_cube.scale_factor
            self._pose_tensor[:3] -= world_cube.shift
        else:
            self._pose_tensor[:3] += world_cube.shift
            self._pose_tensor[:3] /= world_cube.scale_factor

        return self

    def clone(self, fixed=None):
        if fixed is None:
            fixed = self.fixed
        return Pose(pose_tensor=self._pose_tensor.clone(), fixed=fixed)

    def __mul__(self, other) -> "Pose":
        return Pose(self.get_transformation_matrix() @ other.get_transformation_matrix())

    def inv(self) -> "Pose":
        return Pose(torch.linalg.inv(self.get_transformation_matrix()))

    def set_fixed(self, fixed: bool = True) -> None:
        self._pose_tensor.requires_grad_(not fixed)

    def get_transformation_matrix(self):
        return tensor_to_transform(self._pose_tensor)

    def get_pose_tensor(self):
        return self._pose_tensor

    def get_translation(self):
        return self._pose_tensor[:3]

    def get_rotation(self, real_first = False):
        if real_first:
            return torch.cat([self._pose_tensor[-1], self._pose_tensor[3:6]])
        return self._pose_tensor[3:]

    def get_axis_angle(self):
        pytorch3d.transforms.quaternion_to_axis_angle(self.get_rotation(True))
