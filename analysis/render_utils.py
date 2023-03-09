import cv2
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity, equalize_adapthist, equalize_hist
from common.settings import Settings
from matplotlib import cm
from tqdm import tqdm



def vis_flow(flow, scale=0):
    fx, fy = cv2.split(flow)
    mag,ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    if scale== 0:
        cv2.normalize(mag, mag, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    else:
        mag /= scale
    ret = cv2.merge([ang,mag,np.ones_like(mag)])
    ret = cv2.cvtColor(ret, cv2.COLOR_HSV2RGB)
    return ret


def save_video(fname, tensor_list, shape, rescale, clahe, isdepth, cmap='jet', camera_range = None, **kwargs):
    frames = []
    frames_tensor = torch.stack(tensor_list, dim=0).reshape([len(tensor_list)] + list(shape))
    frames = frames_tensor.detach().cpu().numpy()
    frames[np.isnan(frames)] = 0

    if isdepth:
        ## clip depth to camera depth range
        if camera_range:
            frames = np.clip(frames, camera_range[0], camera_range[1])

        ## normalize image to 1, 99 percentile
        min_val = np.percentile(frames, 1)
        max_val = np.percentile(frames, 99)
        frames = (frames-min_val)/(max_val-min_val)
        frames = np.clip(frames, 0, 1)

        ## get colormap for depth visualization
        cm = plt.cm.get_cmap(cmap)
        frames = np.array([cm(frame) for frame in frames])
    else:
        if rescale:
            frames = np.array([rescale_intensity(frame) for frame in frames])
        if clahe:
            frames = np.array([equalize_adapthist(frame, clip_limit=1.0) for frame in frames])
    ## convert to uint8 and save
    frames = frames * 255.
    frames = frames.astype(np.uint8).squeeze()
    imageio.mimwrite(fname, frames, **kwargs)
    return frames
  
    
def depth_to_warp(depth_map1, depth_map2, K1, T12, K2):
    # Given depth map and intrinsics of camera 1, relative pose to camera 2 and depth map and instrinsic of camera 2
    # computes the corresponding warp in image plane pixel coordinates. Along with the occlusion mask.
    
    assert depth_map1.shape == depth_map2.shape, "Expected depth maps to be of same shape"
    H, W = depth_map1.shape
    U = np.linspace(0, W - 1, num=W, dtype=np.float32)
    V = np.linspace(0, H - 1, num=H, dtype=np.float32)
    UU, VV = np.meshgrid(U, V)
    u1 = UU.reshape(-1)
    v1 = VV.reshape(-1)
    # Note: depthmaps contains np.inf for holes. This includes regions (rays) that are beyond lidar range as well regions
    # that were never scanned by a lidar beam even in close regions. 
    d1 = -depth_map1.reshape(-1)     # open3d rendered depth is negative
    fx, cx, fy, cy = K1[0, 0], K1[0, 2], K1[1, 1], K1[1, 2]
    X_Z = (u1 - cx) / fx
    Y_Z = (v1 - cy) / fy
    Z = d1 / np.sqrt(1 + X_Z**2 + Y_Z**2)
    X = X_Z * Z
    Y = Y_Z * Z
    cam1_wpoints = np.stack([X, Y, Z, np.ones_like(X)], axis=0)
    cam2_wpoints = T12 @ cam1_wpoints
    cam2_wpoints = cam2_wpoints[:3, None, :].T
    imgPts2, _ = cv2.projectPoints(cam2_wpoints, rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=K2, distCoeffs=np.zeros(5))
    warp12 = np.stack([imgPts2[:, 0, 0] - u1, imgPts2[:, 0, 1] - v1], axis=1).reshape(H, W, 2)
    warp12[depth_map1 == np.inf] = 0
    
    # Compute occlusion masks
    # Type 1: depth_map1 was np.inf
    # Type 2: consider 4 pixels bordering projected point
    #         2.A: all four pixels have np.inf in depth_map2
    #         2.B: lowest depth among four pixels is lower than depth inferred from depth_map1 by a threshold
    mask1 = (depth_map1.reshape(-1) == np.inf)
    u_min, v_min = np.hsplit(np.floor(imgPts2[:, 0, :]).astype('int'), 2)
    u_max, v_max = np.hsplit(np.ceil(imgPts2[:, 0, :]).astype('int'), 2)
    u_min = u_min.clip(0, W-1)
    u_max = u_max.clip(0, W-1)
    v_min = v_min.clip(0, H-1)
    v_max = v_max.clip(0, H-1)

    # depth of scene points warped from cam1 in cam2 optical frame 
    depthmap_warped = np.linalg.norm(cam2_wpoints, axis=2)
    # minimum depth at destination pixel locations
    distances2 = np.ones((H * W, 1)) * np.inf
    distances2[~mask1] =  np.concatenate([-depth_map2[v_min[~mask1], u_min[~mask1]],
                            -depth_map2[v_min[~mask1], u_max[~mask1]],
                            -depth_map2[v_max[~mask1], u_min[~mask1]],
                            -depth_map2[v_max[~mask1], u_max[~mask1]]], axis=1).min(axis=1, keepdims=True)
    distances2[mask1] = 1e6
    depthmap_warped[mask1] = -1e6
    mask = np.abs(distances2 - depthmap_warped) < 0.5
    mask = mask.reshape(H, W)
    return warp12, mask[..., None]

def save_img(rgb_fine, mask, filename,render_dir,  equalize=False):
    img = rgb_fine.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    if equalize:
        eq_img = equalize_adapthist(img, clip_limit=1.0) * 255
    else:
        eq_img = img * 255
    eq_img[mask, :] = 255
    out_fname = render_dir / f'{filename}'
    imageio.imwrite(str(out_fname), eq_img)

def save_depth(depth_fine, fname, render_dir, min_depth=1, max_depth=50):
    img = depth_fine.squeeze().detach()
    mask = (img >= 50)
    img = torch.clip(img, min_depth, max_depth)
    # img = (img - img.min()) / (np.percentile(img, 99) - img.min())
    img = (img - min_depth) / (max_depth - min_depth)
    img = torch.clip(img, 0, 1).cpu().numpy()
    cmap = plt.cm.get_cmap('turbo')
    img_colored = cmap(img)
    out_fname = render_dir / fname
    imageio.imwrite(str(out_fname), img_colored * 255)