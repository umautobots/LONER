import torch
import scipy.spatial.transform as sptransform
import numpy as np

def transform_to_tensor(transformation_matrix, device=None):
    """
    Converts a homogenous transformation matrix into a tensor T = [x,y,z, x',y',z']
    where x',y',z' are the imaginary components of a quaternion representing the rotation.
    The real component is left implicit so we can implicitly enforce the norm to be 1.c

    Taken from: https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/src/common.py#L179
    """

    gpu_id = -1
    if isinstance(transformation_matrix, torch.Tensor):
        if transformation_matrix.get_device() != -1:
            transformation_matrix = transformation_matrix.detatch().cpu()
            gpu_id = transformation_matrix.get_device()
        transformation_matrix = transformation_matrix.numpy()
    elif not isinstance(transformation_matrix, (np.array, np.ndarray)):
        raise ValueError((f"Invalid argument of type {type(transformation_matrix).__name__}"
                           "passed to transform_to_tensor (Expected numpy array or pytorch tensor)"))

    R = transformation_matrix[:3, :3]
    T = transformation_matrix[:3, 3]

    rotation = sptransform.Rotation.from_matrix(R)
    quat = rotation.as_quat()

    tensor = torch.from_numpy(np.concatenate([T, quat])).float()
    if device is not None:
        tensor = tensor.to(device)
    elif gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def quaternion_to_rotation(quaternion_tensors):
    """
    Converts a quaternion to a tensor using purely pytorch operations, thus preserving gradients.

    Taken from Nice-SLAM implementation.
    https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/src/common.py#L137
    """

    bs = quaternion_tensors.shape[0]

    if len(quaternion_tensors.shape) == 1:
        quaternion_tensors = torch.unsqueeze(quaternion_tensors, 0)
        bs = quaternion_tensors.shape[0]
    
    qi, qj, qk, qr = quaternion_tensors[:, 0], quaternion_tensors[:, 1], quaternion_tensors[:, 2], quaternion_tensors[:, 3]
    two_s = 2.0 / (quaternion_tensors * quaternion_tensors).sum(-1)
    rotation_matrix = torch.zeros(bs, 3, 3)
    if quaternion_tensors.get_device() != -1:
        rotation_matrix = rotation_matrix.to(quaternion_tensors.get_device())
    rotation_matrix[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rotation_matrix[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rotation_matrix[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rotation_matrix[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rotation_matrix[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rotation_matrix[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rotation_matrix[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rotation_matrix[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rotation_matrix[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rotation_matrix


def tensor_to_transform(transformation_tensors):
    """
    Converts a tensor produced by transform_to_tensor to a transformation matrix, of type
    Tensor. This is implemented purely in pytorch, and hence gradients are supported.

    Taken from: https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/src/common.py#L163
    """
    N = len(transformation_tensors.shape)
    if N == 1:
        transformation_tensors = torch.unsqueeze(transformation_tensors, 0)
    Ts,quats = transformation_tensors[:,:3], transformation_tensors[:,3:]
    rotation_matrices = quaternion_to_rotation(quats)
    RT = torch.cat([rotation_matrices, Ts[:,:,None]], 2)
    if N == 1:
        RT = RT[0]
    return RT

class Pose:
    def __init__(self, transformation_matrix: torch.Tensor,
                       fixed: bool=False):
        self._pose_tensor = transform_to_tensor(transformation_matrix)
        self._pose_tensor.requires_grad_(not fixed)

    def FromSettings(pose_dict:dict, fixed: bool=False) -> "Pose":
        xyz = torch.Tensor(pose_dict['xyz'])
        quat = torch.Tensor(pose_dict['orientation'])

        # TODO: This is a very rounabout, and expensive.
        tensor = torch.cat((xyz, quat))
        tf = tensor_to_transform(tensor)
        return Pose(tf, fixed)

    def SetFixed(self, fixed: bool = True) -> None:
        self._pose_tensor.requires_grad_(not fixed)

    def GetTransformationMatrix(self):
        return tensor_to_transform(self._pose_tensor)

    def GetPoseTensor(self):
        return self._pose_tensor