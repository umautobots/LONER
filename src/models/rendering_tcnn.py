import torch


# ref: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    # (N_rays, N_samples_)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    # Take uniform samples
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        # generates random numbers in the interval [0, 1)
        u = torch.rand(N_rays, N_importance, device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1]-cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom[denom < eps] = 1
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u-cdf_g[..., 0]) / \
        denom * (bins_g[..., 1]-bins_g[..., 0])
    return samples


# ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py#L262
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, sigma_only=False, num_colors=3, softplus=False, far=None, ret_var=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4(sigma_only=False) or 1(sigma_only=True)]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        sigma_only: Only sigma was predicted by network
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    if sigma_only:
        sigmas = raw[..., 0]  # (N_rays, N_samples_)
    else:
        rgbs = raw[..., :num_colors]  # (N_rays, N_samples_, 3)
        sigmas = raw[..., num_colors]  # (N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # (N_rays, 1) the last delta is infinity
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(rays_d.unsqueeze(1), dim=-1)

    noise = 0.
    if raw_noise_std > 0:
        noise = torch.randn(sigmas.shape, device=sigmas.device) * raw_noise_std

    # compute alpha by the formula (3)
    if softplus == True:
        # (N_rays, N_samples_)
        alphas = 1 - torch.exp(-deltas*torch.nn.functional.softplus(sigmas+noise))
    else:
        # (N_rays, N_samples_)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise))

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1. -
                  alphas+1e-10], -1)  # [1, a1, a2, ...]
    # (N_rays, N_samples_)
    weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]
    # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    opacity_map = torch.sum(weights, -1)
    # weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # compute final weighted outputs
    if far is not None:
        z_vals_appended = torch.cat([z_vals, far], dim=-1)
        weights_appended = torch.cat(
            [weights, 1-weights.sum(dim=1, keepdim=True)], dim=1)
        depths = torch.sum(weights_appended*z_vals_appended, -1)
    else:
        depths = torch.sum(weights*z_vals, -1) 

    if sigma_only:
        rgb_map = torch.tensor([-1.])
    else:
        # weights_normed = weights / (weights.sum(1, keepdim=True) + 1e-6)
        rgb_map = torch.sum(weights.unsqueeze(-1)*rgbs, -2)  # (N_rays, 3)

        if white_bkgd:
            rgb_map = rgb_map + 1-weights.sum(-1, keepdim=True)

    if ret_var:
        variance = (weights * (depths.view(-1, 1) - z_vals)**2).sum(dim=1)
        return rgb_map, depths, weights, opacity_map, variance

    return rgb_map, depths, weights, opacity_map, None

def inference(model, xyz_, dir_, sigma_only=False, netchunk=32768, detach_sigma=True, meshing=False):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model instantiated using Tiny Cuda NN
        xyz_: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray;
        dir_: (N_rays, 3) ray directions
        sigma_only: do inference on sigma only or not
    Outputs:
        if sigma_only:
            raw: (N_rays, N_samples_, 1): predictions of each sample
        else:
            raw: (N_rays, N_samples_, num_colors + 1): predictions of each sample
    """
    N_rays, N_samples_ = xyz_.shape[0:2]
    xyz_ = xyz_.view(-1, 3).contiguous()  # (N_rays*N_samples_, 3)
    if sigma_only:
        dir_ = None
    else:
        # (N_rays*N_samples_, embed_dir_channels)
        dir_ = torch.repeat_interleave(
            dir_, repeats=N_samples_, dim=0).contiguous()

    # Perform model inference to get color and raw sigma
    B = xyz_.shape[0]
    if netchunk == 0:
        out = model(xyz_, dir_, sigma_only, detach_sigma)
    else:
        out_chunks = []
        for i in range(0, B, netchunk):
            out_chunks += [model(xyz_, dir_, sigma_only, detach_sigma)]
        out = torch.cat(out_chunks, 0)

    if meshing:
        return out
    else:
        return out.view(N_rays, N_samples_, -1)


# Use Fully Fused MLP from Tiny CUDA NN
# Volumetric rendering
def render_rays(rays,
                ray_sampler,
                nerf_model,
                ray_range,
                scale_factor,
                N_samples=64,
                retraw=False,
                perturb=0,
                white_bkgd=False,
                raw_noise_std=0.,
                netchunk=32768,
                num_colors=3,
                sigma_only=False,
                DEBUG=False,
                detach_sigma=True,
                return_variance = False
                ):
    """
    Render rays by computing the output of @occ_model, sampling points based on the class probabilities, applying volumetric rendering using @tcnn_model applied on sampled points

    Inputs:
        rays: (N_rays, 3+3+3+2+?), ray origins, ray directions, unit-magnitude viewing direction, pixel coordinates, ray bin centers
        models: Occupancy Model and NeRF model instantiated using tinycudann
        N_samples: number of samples per ray
        retraw: bool. If True, include model's raw unprocessed predictions.
        lindisp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray
        white_bkgd: whether the background is white (dataset dependent)
        raw_noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final color and depth maps
            color_map: [num_rays, 3 or 1]. Estimated color of a ray.
            depth_map: [num_rays].
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray.
            raw: [num_rays, num_samples, 4 or 1]. Raw predictions from model.
    """
    
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    viewdirs = rays[:, 6:9]  # (N_rays, 3)
    near = rays[:, -2:-1]
    far = rays[:, -1:]

    z_vals = ray_sampler.get_samples(rays, N_samples, perturb)

    xyz_samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    raw = inference(nerf_model, xyz_samples, viewdirs, netchunk=netchunk, sigma_only=sigma_only, detach_sigma=detach_sigma)

    rgb, depth, weights, opacity, variance = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, sigma_only=sigma_only, num_colors=num_colors, far=far, ret_var=return_variance)

    result = {'rgb_fine': rgb,
              'depth_fine': depth,
              'weights_fine': weights,
              'opacity_fine': opacity,
              }

    if return_variance:
        result["variance"] = variance

    if retraw:
        result['samples_fine'] = z_vals
        result['points_fine'] = xyz_samples

    if DEBUG:
        result['raw_fine'] = raw

        for k in result:
            if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
    return result