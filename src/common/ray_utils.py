import struct

import numpy as np
import torch
import open3d as o3d
from kornia.geometry.calibration import undistort_points

from common.pose import Pose
from common.sensors import Image
from common.settings import Settings
from common.pose_utils import WorldCube


# For each dimension, answers the question "how many unit vectors do we need to go
# until we exit the unit cube in this axis."
# i.e. at (0,0) with unit vector (0.7, 0.7), the result should be (1.4, 1.4)
# since after 1.4 unit vectors we're at the exit of the world cube.  
def get_far_val(pts_o: torch.Tensor, pts_d: torch.Tensor, no_nan: bool = False):

    #TODO: This is a hack
    if no_nan:
        pts_d = pts_d + 1e-15

    # Intersection with z = -1, z = 1
    t_z1 = (-1 - pts_o[:, [2]]) / pts_d[:, [2]]
    t_z2 = (1 - pts_o[:, [2]]) / pts_d[:, [2]]
    # Intersection with y = -1, y = 1
    t_y1 = (-1 - pts_o[:, [1]]) / pts_d[:, [1]]
    t_y2 = (1 - pts_o[:, [1]]) / pts_d[:, [1]]
    # Intersection with x = -1, x = 1
    t_x1 = (-1 - pts_o[:, [0]]) / pts_d[:, [0]]
    t_x2 = (1 - pts_o[:, [0]]) / pts_d[:, [0]]

    clipped_ts = torch.cat([torch.maximum(t_z1.clamp(min=0), t_z2.clamp(min=0)),
                            torch.maximum(t_y1.clamp(min=0),
                                          t_y2.clamp(min=0)),
                            torch.maximum(t_x1.clamp(min=0), t_x2.clamp(min=0))], dim=1)
    far_clip = clipped_ts.min(dim=1)[0].unsqueeze(1)
    return far_clip

def get_ray_directions(H, W, newK, dist=None, K=None, sppd=1, with_indices=False):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    When images are already undistorted and rectified, only use newK.
    When images are unrectified, we compute the pixel locations in the undistorted image plane
    and use that to compute ray directions as though a camera with newK intrinsic matrix was used
    to capture the scene. 
    Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a

    Inputs:
        H, W: image height and width
        K: intrinsic matrix of the camera
        dist: distortion coefficients
        newK: output of getOptimalNewCameraMatrix. Compute with free scaling parameter set to 1. 
        This retains all pixels in the undistorted image.         
        sppd: samples per pixel along each dimension. Returns sppd**2 number of rays per pixel
        with_indices: if True, additionally return the i and j meshgrid indices
    Outputs:
        directions: (H*W, 3), the direction of the rays in camera coordinate
        i (optionally) : (H*W, 1) integer pixel coordinate
        j (optionally) : (H*W, 1) integer pixel coordinate
    """
    # offset added so that rays emanate from around center of pixel
    xs = torch.linspace(0, W - 1. / sppd, sppd * W, device=newK.device)
    ys = torch.linspace(0, H - 1. / sppd, sppd * H, device=newK.device)

    grid_x, grid_y = torch.meshgrid([xs, ys])
    grid_x = grid_x.permute(1, 0).reshape(-1, 1)    # (H*W, 1)
    grid_y = grid_y.permute(1, 0).reshape(-1, 1)    # (H*W, 1)

    if dist is not None:
        assert K is not None

        # computing the undistorted pixel locations
        print(f"Computing the undistorted pixel locations to compute the correct ray directions")

        if len(K.shape) == 2:
            K = K.unsqueeze(0)  # (1, 3, 3)

        if isinstance(dist, list):
            dist = torch.tensor(dist, device=K.device).unsqueeze(0)  # (1, 5)

        assert newK is not None, f"Compute newK using getOptimalNewCameraMatrix with alpha=1"
        if len(newK.shape) == 2:
            newK = newK.unsqueeze(0)  # (1, 3, 3)

        points = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)
        undistorted_points = undistort_points(points, K, dist, newK)
        new_grid_x = undistorted_points[0, :, 0:1]
        new_grid_y = undistorted_points[0, :, 1:2]
        newK = newK[0]
    else:
        new_grid_x = grid_x
        new_grid_y = grid_y

    directions = torch.cat([(new_grid_x-newK[0, 2])/newK[0, 0], -(
        new_grid_y-newK[1, 2])/newK[1, 1], -torch.ones_like(grid_x)], -1)  # (H*W, 3)

    if with_indices:
        # returns directions computed from undistorted pixel locations along with the pixel locations in the original distorted image
        return directions, grid_x, grid_y
    else:
        return directions


class CameraRayDirections:
    def __init__(self, calibration: Settings, samples_per_pixel: int = 1, device = 'cpu', chunk_size=512,
                 grid_dimensions = (8,8), samples_per_grid_cell = 12):

        K = calibration.camera_intrinsic.k.to(device)
        distortion = calibration.camera_intrinsic.distortion.to(device)
        new_k = calibration.camera_intrinsic.new_k

        if new_k is None:
            print("Warning: No New K provided. Using K")
            new_k = K
        
        new_k = new_k.to(device)

        self.im_width = calibration.camera_intrinsic.width
        self.im_height = calibration.camera_intrinsic.height

        self.directions, self.i_meshgrid, self.j_meshgrid = get_ray_directions(
            self.im_height, self.im_width, newK=new_k, dist=distortion,
            K=K, sppd=samples_per_pixel, with_indices=True)

        self.directions = self.directions.to(device)
        self.i_meshgrid = self.i_meshgrid.to(device)
        self.j_meshgrid = self.j_meshgrid.to(device)

        self._chunk_size = chunk_size
        self.num_chunks = int(np.ceil(self.directions.shape[0] / self._chunk_size))

        self.grid_dimensions = grid_dimensions
        self.samples_per_grid_cell = samples_per_grid_cell
        self.grid_cell_width = int(self.im_width // self.grid_dimensions[1])
        self.grid_cell_height = int(self.im_height // self.grid_dimensions[0])
        self.total_grid_samples = self.grid_dimensions[0]*self.grid_dimensions[1]*samples_per_grid_cell
        x_cell_offsets = torch.arange(0, self.im_width, int(self.im_width // self.grid_dimensions[1]))
        y_cell_offsets = torch.arange(0, self.im_height, int(self.im_height // self.grid_dimensions[0]))

        x_offsets_grid, y_offsets_grid =torch.meshgrid([x_cell_offsets, y_cell_offsets])

        x_cell_offsets = x_offsets_grid.permute(1,0).reshape(-1,1)
        y_cell_offsets = y_offsets_grid.permute(1,0).reshape(-1,1)

        # Stored in row-major order
        self.cell_offsets = torch.hstack((y_cell_offsets,x_cell_offsets)).unsqueeze(1).to(torch.int32)
        
    def __len__(self):
        return self.directions.shape[0]

    def build_rays(self, camera_indices: torch.Tensor, pose: Pose, image: Image, world_cube: WorldCube, ray_range):

        directions = self.directions[camera_indices]
        ray_i_grid = self.i_meshgrid[camera_indices]
        ray_j_grid = self.j_meshgrid[camera_indices]

        # T_camera_opengl = torch.Tensor([[1, 0, 0, 0],
        #                                 [0, -1, 0, 0],
        #                                 [0, 0, -1, 0],
        #                                 [0, 0, 0, 1]]).to(directions.device)

        T_camera_opengl = torch.eye(4).to(directions.device)
                
        world_to_camera_opengl = pose.get_transformation_matrix() @ T_camera_opengl

        # Now that we're in OpenGL frame, we can apply world cube transformation
        world_to_camera_opengl[:3, 3] = world_to_camera_opengl[:3, 3] + world_cube.shift
        world_to_camera_opengl[:3, 3] = world_to_camera_opengl[:3, 3] / world_cube.scale_factor
        
        
        ray_directions = directions @ world_to_camera_opengl[:3, :3].T
        # Note to self: don't use /= here. Breaks autograd.
        ray_directions = ray_directions / \
            torch.norm(ray_directions, dim=-1, keepdim=True)
        ray_directions = ray_directions[:, :3]

        ray_origins = torch.zeros_like(ray_directions)
        ray_origins_homo = torch.cat(
            [ray_origins, torch.ones_like(ray_origins[:, :1])], dim=-1)
        ray_origins = ray_origins_homo @ world_to_camera_opengl[:3, :].T
        ray_origins = ray_origins

        view_directions = -ray_directions
        near = ray_range[0] / world_cube.scale_factor * \
            torch.ones_like(ray_origins[:, :1])

        # TODO: Does no_nan cause problems here?
        far = get_far_val(ray_origins, ray_directions, no_nan=True)
        rays = torch.cat([ray_origins, ray_directions, view_directions,
                            ray_i_grid, ray_j_grid, near, far], 1).float()

        if image is not None:
            # Get intensities
            img = image.image

            # TODO: Handle distortion and sppd > 1
            intensities = img.view(-1, img.shape[2])[camera_indices]
        else:
            intensities = None

        return rays, intensities

    def fetch_chunk_rays(self, chunk_idx: int, pose: Pose, world_cube, ray_range):
        start_idx = chunk_idx*self._chunk_size
        end_idx = min(self.directions.shape[0], (chunk_idx+1)*self._chunk_size)
        indices = torch.arange(start_idx, end_idx, 1)

        return self.build_rays(indices, pose, None, world_cube, ray_range)[0]

    # Sample distribution is 1D, in row-major order (size of grid_dimensions)
    def sample_chunks(self, sample_distribution: torch.Tensor = None, total_grid_samples = None) -> torch.Tensor:
        
        if total_grid_samples is None:
            total_grid_samples = self.total_grid_samples

        # TODO: This method potentially ignores the upper-right border of each cell. fixme.
        num_grid_cells = self.grid_dimensions[0]*self.grid_dimensions[1]
        
        if sample_distribution is None:
            local_xs = torch.randint(0, self.grid_cell_width, (total_grid_samples,))
            local_ys = torch.randint(0, self.grid_cell_height, (total_grid_samples,))

            local_xs = local_xs.reshape(num_grid_cells, self.samples_per_grid_cell, 1)
            local_ys = local_ys.reshape(num_grid_cells, self.samples_per_grid_cell, 1)

            # num_grid_cells x samples_per_grid_cell x 2
            local_samples = torch.cat((local_ys, local_xs), dim=2)

            # Row-major order
            samples = local_samples + self.cell_offsets

            indices = samples[:,:,0]*self.im_width + samples[:,:,1]
        else:
            local_xs = torch.randint(0, self.grid_cell_width, (total_grid_samples,))
            local_ys = torch.randint(0, self.grid_cell_height, (total_grid_samples,))     
            all_samples = torch.vstack((local_ys, local_xs)).T

            # TODO: There must be a better way
            samples_per_cell: torch.Tensor = sample_distribution * total_grid_samples
            samples_per_cell = samples_per_cell.floor().to(torch.int32)
            remainder = total_grid_samples - samples_per_cell.sum()

            while remainder > len(samples_per_cell):
                samples_per_cell += 1
                remainder -= len(samples_per_cell)

            _, best_indices = samples_per_cell.topk(remainder)
            samples_per_cell[best_indices] += 1
            

            repeated_cell_offsets = self.cell_offsets.squeeze(1).repeat_interleave(samples_per_cell, dim=0)
            all_samples += repeated_cell_offsets
            
            indices = all_samples[:,0]*self.im_width + all_samples[:,1]

        return indices


def rays_to_o3d(rays, depths, intensities=None):
    origins = rays[:, :3]
    directions = rays[:, 3:6]
    
    depths = depths.reshape((depths.shape[0], 1))
    end_points = origins + directions*depths

    end_points = end_points.detach().cpu().numpy()

    pcd = o3d.cuda.pybind.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(end_points)

    if intensities is not None:
        intensities = intensities.detach().cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(intensities)

    return pcd

def rays_to_pcd(rays, depths, rays_fname, origins_fname, intensities=None):

    if intensities is None:
        intensities = torch.ones_like(rays[:, :3])
    

    intensity_floats = []
    for intensity_row in intensities:
        red = int(intensity_row[0] * 255).to_bytes(1, 'big', signed=False)
        green = int(intensity_row[1] * 255).to_bytes(1, 'big', signed=False)
        blue = int(intensity_row[2] * 255).to_bytes(1, 'big', signed=False)

        intensity_bytes = struct.pack("4c", red, green, blue, b"\x00")
        intensity_floats.append(struct.unpack('f', intensity_bytes)[0])


    origins = rays[:, :3]
    directions = rays[:, 3:6]
    
    depths = depths.reshape((depths.shape[0], 1))
    end_points = origins + directions*depths
        
    with open(rays_fname, 'w') as f:
        if end_points.shape[0] <= 3:
            end_points = end_points.T
            assert end_points.shape[0] > 3, f"Too few points or wrong shape of pcd file."
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {end_points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {end_points.shape[0]}\n")
        f.write("DATA ascii\n")
        for pt, intensity in zip(end_points, intensity_floats):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {intensity}\n")

    with open(origins_fname, 'w') as f:
        if origins.shape[0] <= 3:
            origins = origins.T
            assert origins.shape[0] > 3, f"Too few points or wrong shape of pcd file."
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4 \n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {origins.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {origins.shape[0]}\n")
        f.write("DATA ascii\n")
        for pt in origins:
            f.write(f"{pt[0]} {pt[1]} {pt[2]} \n")