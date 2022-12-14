import torch
from common.settings import Settings
from kornia.geometry.calibration import undistort_points

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
    def __init__(self, calibration: Settings, samples_per_pixel: int = 1):

        K = calibration.camera_intrinsic.k
        distortion = calibration.camera_intrinsic.distortion
        new_k = calibration.camera_intrinsic.new_k

        if new_k is None:
            print("Warning: No New K provided. Using K")
            new_k = K

        im_width = calibration.camera_intrinsic.width
        im_height = calibration.camera_intrinsic.height

        self.directions, self.i_meshgrid, self.j_meshgrid = get_ray_directions(
            im_height, im_width, newK=new_k, dist=distortion,
            K=K, sppd=samples_per_pixel, with_indices=True)
