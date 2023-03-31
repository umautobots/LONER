# Several functions have been adapted from: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
from dataclasses import dataclass

import numpy as np
import pytorch3d.transforms
import torch
import pandas as pd
from scipy.spatial.transform import Rotation


@dataclass
class WorldCube:
    """
    The WorldCube struct holds a shift and scale transformation to apply to poses 
    before creating rays, such that all rays are within a unit-length cube.
    """
    
    scale_factor: torch.Tensor
    shift: torch.Tensor

    def to(self, device, clone=False) -> "WorldCube":

        if clone:

            if isinstance(self.shift, torch.Tensor):
                shift = self.shift.to(device, copy=True)
            else:
                shift = torch.Tensor([self.shift], device)
            scale_factor = self.scale_factor.to(device, copy=True)
            return WorldCube(scale_factor, shift)

        if isinstance(self.shift, torch.Tensor):
            self.shift = self.shift.to(device)
        else:
            self.shift = torch.Tensor([self.shift], device)

        self.scale_factor = self.scale_factor.to(device)
        return self

    def as_dict(self) -> dict:
        shift = [float(s) for s in self.shift.cpu()]
        return {
            "scale_factor": float(self.scale_factor.cpu()),
            "shift": shift
        }

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
    # Note (seth): This is a hack around pylance.
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

## Compute an axis aligned minimal cube encompassing sensor poses and camera view frustums with 
# the given camera range. An additional padding is added. 
# @param camera_to_lidar: The extrinsic calibration
# @param intrinsic_mats: Either one 3x3 intrinsic matrix, or one matrix per pose.
# @param image_sizes: Either one tuple/tensor specifying image dimensions, or one per pose
# @param lidar_poses: Groundtruth lidar poses
# @param ray_range: A tuple with min and max camera range
# @param padding: 0 means no extra padding is added, the cube is doubled in each axis
def compute_world_cube(camera_to_lidar, intrinsic_mats, image_sizes, lidar_poses, ray_range, padding=0.1, traj_bounding_box=None) -> WorldCube:

    assert 0 <= padding < 1
    
    assert lidar_poses is not None or traj_bounding_box is not None

    if lidar_poses is None:
        print("Computing world cube using supplied trajectory bounding box")
        x_min, x_max = traj_bounding_box['x']
        y_min, y_max = traj_bounding_box['y']
        z_min, z_max = traj_bounding_box['z']

        x_range = torch.Tensor([x_min, x_max])
        y_range = torch.Tensor([y_min, y_max])
        z_range = torch.Tensor([z_min, z_max])

        all_combos = torch.stack(torch.meshgrid([x_range, y_range, z_range]), dim=-1).reshape(-1, 3, 1)
        
        lidar_poses = torch.eye(4).tile((8, 1, 1))
        
        lidar_poses[:,:3,3:4] = all_combos
    else:
        print("Computing world cube with groundtruth poses")
        lidar_poses = lidar_poses @ lidar_poses[0,:,:].inverse()

    if camera_to_lidar is None:
        camera_poses = []
    else:
        camera_poses = lidar_poses @ camera_to_lidar.inverse()

    
    camera_poses = camera_poses
    lidar_poses = lidar_poses


    all_corners = []
    
    if camera_to_lidar is not None:
        if len(intrinsic_mats.shape) == 2:
            intrinsic_mats = torch.broadcast_to(
                intrinsic_mats, (camera_poses.shape[0], 3, 3))
        if isinstance(image_sizes, tuple):
            image_sizes = torch.Tensor(image_sizes)
        if image_sizes.shape == (2,):
            image_sizes = torch.broadcast_to(
                image_sizes, (camera_poses.shape[0], 2))
        else:
            assert image_sizes.shape[0] == camera_poses.shape[0]
            
        for K, hw, c2w in zip(intrinsic_mats, image_sizes, camera_poses):
            pts_homo = _get_view_frustum_corners(
                K, hw[0], hw[1], min_depth=ray_range[0], max_depth=ray_range[1])    # (8, 4)
            corners = c2w[:3, :] @ pts_homo.T
            all_corners += [corners.T]


        all_corners = torch.cat(all_corners, dim=0)  # (8N, 3)
        all_poses = torch.cat(
            [camera_poses[..., :3, 3], lidar_poses[..., :3, 3]], dim=0)

    else:
        max_depth = ray_range[1]
        lidar_view_corners = torch.Tensor([[-max_depth, -max_depth, -max_depth, 1],
                                        [-max_depth, max_depth, -max_depth, 1],
                                        [max_depth, -max_depth, -max_depth, 1],
                                        [max_depth, max_depth, -max_depth, 1],
                                        [-max_depth, -max_depth, max_depth, 1],
                                        [-max_depth, max_depth, max_depth, 1],
                                        [max_depth, -max_depth, max_depth, 1],
                                        [max_depth, max_depth, max_depth, 1]])

        for c2l in lidar_poses:
            corners = c2l[:3,:] @ lidar_view_corners.T
            all_corners += [corners.T]

        all_corners = torch.cat(all_corners, dim=0)

        all_poses = lidar_poses[...,:3,3]

    all_points = torch.cat([all_corners, all_poses])

    min_coord = all_points.min(dim=0)[0]
    max_coord = all_points.max(dim=0)[0]

    origin = min_coord + (max_coord - min_coord) / 2

    scale_factor = (torch.linalg.norm(max_coord - min_coord) /
                    (2 * torch.sqrt(torch.Tensor([3])))) * (1+padding)

    return WorldCube(scale_factor, -origin)

## Converts a 4x4 transformation matrix to the se(3) twist vector
# Inspired by a similar NICE-SLAM function.
# @param transformation_matrix: A pytorch 4x4 homogenous transformation matrix
# @param device: The device for the output
# @returns: A 6-tensor [x,y,z,r_x,r_y,r_z]
def transform_to_tensor(transformation_matrix, device=None):

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

    rot = pytorch3d.transforms.matrix_to_axis_angle(R)

    tensor = torch.cat([T, rot]).float()
    if device is not None:
        tensor = tensor.to(device)
    elif gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


## Converts a tensor produced by transform_to_tensor to a transformation matrix
# Inspired by a similar NICE-SLAM function.
# @param transformation_tensors: se(3) twist vectors
# @returns a 4x4 homogenous transformation matrix
def tensor_to_transform(transformation_tensors):

    N = len(transformation_tensors.shape)
    if N == 1:
        transformation_tensors = torch.unsqueeze(transformation_tensors, 0)
    Ts, rots = transformation_tensors[:, :3], transformation_tensors[:, 3:]
    rotation_matrices = pytorch3d.transforms.axis_angle_to_matrix(rots)
    RT = torch.cat([rotation_matrices, Ts[:, :, None]], 2)
    if N == 1:
        RT = RT[0]

    H_row = torch.zeros_like(RT[0])
    H_row[3] = 1
    RT = torch.vstack((RT, H_row))
    return RT

## Given a set of transformation matrices and timestamps, dumps the trajectory to TUM format.
# @param transformation_matrices: Nx4x4 homogenous transforms representing the poses
# @param timestamps: N timestamps, one per pose
# @param output_file: path to dump result to.
def dump_trajectory_to_tum(transformation_matrices: torch.Tensor,
                      timestamps: torch.Tensor,
                      output_file: str) -> None:
    
    translations = transformation_matrices[:, :3, 3].reshape(-1, 3)
    rotations = pytorch3d.transforms.matrix_to_quaternion(transformation_matrices[:, :3, :3]).reshape(-1, 4)
    # swap w,x,y,z to x,y,z,w
    rotations = torch.hstack([rotations[:,1:4], rotations[:, 0:1]])
    data = torch.hstack([timestamps.reshape(-1,1), translations, rotations])
    data = data.detach().cpu().numpy()
    np.savetxt(output_file, data, delimiter=" ", fmt="%.10f")


def build_poses_from_df(df: pd.DataFrame, zero_origin=False):
    data = torch.from_numpy(df.to_numpy(dtype=np.float64))

    ts = data[:,0]
    xyz = data[:,1:4]
    quat = data[:,4:]

    rots = torch.from_numpy(Rotation.from_quat(quat).as_matrix())
    
    poses = torch.cat((rots, xyz.unsqueeze(2)), dim=2)

    homog = torch.Tensor([0,0,0,1]).tile((poses.shape[0], 1, 1)).to(poses.device)

    poses = torch.cat((poses, homog), dim=1)

    if zero_origin:
        rot_inv = poses[0,:3,:3].T
        t_inv = -rot_inv @ poses[0,:3,3]
        start_inv = torch.hstack((rot_inv, t_inv.reshape(-1, 1)))
        start_inv = torch.vstack((start_inv, torch.tensor([0,0,0,1.0], device=start_inv.device)))
        poses = start_inv.unsqueeze(0) @ poses

    return poses.float(), ts