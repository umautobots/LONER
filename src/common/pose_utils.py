# Several functions have been adapted from: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
from dataclasses import dataclass
import numpy as np
import torch
import scipy.spatial.transform as sptransform

@dataclass
class WorldCube:
    scale_factor: float
    shift: float


def normalize(v):
    """Normalize a vector."""
    return v/torch.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    # Note (seth): Don't push this. Hack around pylance.
    def do_cross(x, y): return torch.cross(x, y)
    x = normalize(do_cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = do_cross(z, x)  # (3)

    pose_avg = torch.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (4, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = torch.eye(4)
    # convert to homogeneous coordinate for faster computation
    pose_avg_homo[:3] = pose_avg
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = torch.tile(torch.Tensor([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    # (N_images, 4, 4) homogeneous coordinate
    poses_homo = torch.cat([poses, last_row], 1)

    poses_centered = torch.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, torch.linalg.inv(pose_avg_homo)


def _get_view_frustum_corners(K, H, W, min_depth=1, max_depth=1e6):
    assert min_depth < max_depth
    assert min_depth > 0 and max_depth > 0
    return torch.Tensor([[-K[0, 2] / K[0, 0] * min_depth, K[1, 2] / K[1, 1] * min_depth, -min_depth, 1],          # left, up, near
                    [-K[0, 2] / K[0, 0] * max_depth, K[1, 2] / K[1, 1] * \
                        max_depth, -max_depth, 1],           # left, up, far
                    # left, down, near
                     [-K[0, 2] / K[0, 0] * min_depth, - \
                         (H-K[1, 2]) / K[1, 1] * min_depth, -min_depth, 1],
                     # left, down, far
                     [-K[0, 2] / K[0, 0] * max_depth, - \
                         (H-K[1, 2]) / K[1, 1] * max_depth, -max_depth, 1],
                     [(W-K[0, 2]) / K[0, 0] * min_depth, K[1, 2] / K[1, 1]
                     * min_depth, -min_depth, 1],        # right, up, near
                     [(W-K[0, 2]) / K[0, 0] * max_depth, K[1, 2] / K[1, 1]
                     * max_depth, -max_depth, 1],        # right, up, far
                     [(W-K[0, 2]) / K[0, 0] * min_depth, -(H-K[1, 2]) / K[1, 1]
                     * min_depth, -min_depth, 1],   # right, down, near
                     [(W-K[0, 2]) / K[0, 0] * max_depth, -(H-K[1, 2]) / K[1, 1] * max_depth, -max_depth, 1]])  # right, down, far


# TODO (seth): Update comments, this is all fake news now. we don't actually move anything
def compute_world_cube(camera_poses, intrinsic_mats, image_sizes, lidar_poses, camera_range, padding=0.3) -> WorldCube:
    """
    Compute an axis aligned minimal cube encompassing sensor poses and camera view frustums with the given camera range. 
    An additional padding is added. 

    The computation is as follows:
    1. For each camera (pose, intrinsic, size), compute the 8 corners of the view frustum in its local coordinate frame.
    2. Transform all the corners into the input-world frame
    3. Project all pointclouds to input-world frame    
    4. Collect all points in input-world frame: all point clouds, all view frustum corners, all sensor positions
    5. Compute an oriented cube encompassing these points.
    6. Transform all the poses relative to the unit cube
    7. Scale the poses down to fit within unit cube. Scale down by an additional factor of (1-padding)
    8. Scale the pointclouds

    Inputs:
        camera_poses: (N, 3, 4)
        intrinsic_mtxs: (3, 3) or (N, 3, 3)
        sizes: (N, 2) or (2,)
        clouds: List of (*, 3) pointclouds
        lidar_poses: (M, 3, 4) 
        camera_far_plane: maximum depth to consider for each camera
        padding: world would be additionally scaled down by a factor of (1-padding)
    Outputs:
        scaled_clouds: List of (*, 3) pointclouds
        lidar_poses_world: (M, 3, 4)
        camera_poses_world: (N, 3, 4)
        shift_world: (3,) transformation to go from input sensor poses to output sensor poses
        scale_factor: float
    """
    assert 0 <= padding < 1

    if len(intrinsic_mats.shape) == 2:
        intrinsic_mats = torch.broadcast_to(
            intrinsic_mats, (camera_poses.shape[0], 3, 3))
    if isinstance(image_sizes, tuple):
        image_sizes = torch.Tensor(image_sizes)
    if image_sizes.shape == (2,):
        image_sizes = torch.broadcast_to(image_sizes, (camera_poses.shape[0], 2))
    else:
        assert image_sizes.shape[0] == camera_poses.shape[0]

    all_corners = []
    for K, hw, c2w in zip(intrinsic_mats, image_sizes, camera_poses):
        # TODO: min_depth expects near plane value. camera_range[0] is the minimum distance along ray
        # need to compute min_depth by looking at the z value of the corner ray with length camera_range[0]
        pts_homo = _get_view_frustum_corners(
            K, hw[0], hw[1], min_depth=camera_range[0], max_depth=camera_range[1])    # (8, 4)
        corners = c2w[:3,:] @ pts_homo.T
        all_corners += [corners.T]
    all_corners = torch.cat(all_corners, dim=0)  # (8N, 3)
    print(f'Computed a list of corners with shape {all_corners.shape}')

    all_poses = torch.cat(
        [camera_poses[...,:3, 3], lidar_poses[...,:3, 3]], dim=0)
    print(f'Computed a list of poses with shape {all_poses.shape}')

    all_points = torch.cat([all_corners, all_poses])
    print(f'Accumulated all points to get: {all_points.shape}')

    # TODO: Convert this to minimal volume bounding box.
    # For now, using Axis Aligned Bounding Box

    min_coord = all_points.min(dim=0)[0]
    max_coord = all_points.max(dim=0)[0]

    # TODO: Change to scaling world cube without shifting origin

    origin = min_coord + (max_coord - min_coord) / 2
    # origin = torch.zeros(3)
    print(
        f'Minimum coordinate: {min_coord}, Maximum coordinate: {max_coord}, New origin: {origin}')

    scale_factor = (torch.linalg.norm(max_coord - min_coord) /
                    (2 * torch.sqrt(torch.Tensor([3])))) * (1+padding)

    return WorldCube(scale_factor, -origin)


def create_spiral_poses(radii, focus_depth, n_poses=60):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """
    poses_spiral = []
    for t in torch.linspace(0, 4*torch.pi, n_poses+1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = torch.Tensor([torch.cos(t), -torch.sin(t), -torch.sin(0.5*t)]) * radii
        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - torch.Tensor([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = torch.Tensor([0, 1, 0])  # (3)
        x = normalize(torch.cross(y_, z))  # (3)
        y = torch.cross(z, x)  # (3)#
        poses_spiral += [torch.stack([x, y, z, center], 1)]  # (3, 4)#
    return torch.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def transform_to_tensor(transformation_matrix, device=None):
    """
    Converts a homogenous transformation matrix into a tensor T = [x,y,z, x',y',z']
    where x',y',z' are the imaginary components of a quaternion representing the rotation.
    The real component is left implicit so we can implicitly enforce the norm to be 1.c

    Taken from: https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/src/common.py#L179
    """

    gpu_id = -1
    if isinstance(transformation_matrix, np.ndarray):
        if transformation_matrix.get_device() != -1:
            if transformation_matrix.requires_grad:
                transformation_matrix = transformation_matrix.detach()
            transformation_matrix = transformation_matrix.detach().cpu()
            gpu_id = transformation_matrix.get_device()
        elif transformation_matrix.requires_grad:
            transformation_matrix = transformation_matrix.detach()
        transformation_matrix = transformation_matrix.numpy()
    elif not isinstance(transformation_matrix, torch.Tensor):
        raise ValueError((f"Invalid argument of type {type(transformation_matrix).__name__}"
                          "passed to transform_to_tensor (Expected numpy array or pytorch tensor)"))

    R = transformation_matrix[:3, :3]
    T = transformation_matrix[:3, 3]

    rotation = sptransform.Rotation.from_matrix(R.detach())
    quat = torch.from_numpy(rotation.as_quat())

    tensor = torch.cat([T, quat]).float()
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

    qi, qj, qk, qr = quaternion_tensors[:, 0], quaternion_tensors[:,
                                                                  1], quaternion_tensors[:, 2], quaternion_tensors[:, 3]
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
    Ts, quats = transformation_tensors[:, :3], transformation_tensors[:, 3:]
    rotation_matrices = quaternion_to_rotation(quats)
    RT = torch.cat([rotation_matrices, Ts[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    RT = torch.vstack((RT, torch.Tensor([0, 0, 0, 1])))
    return RT


