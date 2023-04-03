import torch
from models.model_tcnn import OccupancyGridModel
from models.rendering_tcnn import sample_pdf


class UniformRaySampler():
    def __init__(self):
        print('Initializing a uniform ray sampler')

    def get_samples(self, rays, N_samples, perturb):
        N_rays = rays.shape[0]
        near = rays[:, -2:-1]
        far = rays[:, -1:]
        with torch.no_grad():
            z_steps = torch.linspace(
                0, 1, N_samples, device=rays.device)        # (N_samples)
            # z_steps = torch.logspace(-4, 0, N_samples, device=rays.device)       # (N_samples)
            z_vals = near * (1-z_steps) + far * z_steps
            z_vals = z_vals.expand(N_rays, N_samples)

            if perturb > 0:  # perturb z_vals
                # (N_rays, N_samples-1) interval mid points
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                # get intervals between samples
                upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
                lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
                perturb_rand = perturb * \
                    torch.rand(z_vals.shape, device=rays.device)
                z_vals = lower + (upper - lower) * perturb_rand

        return z_vals


class OccGridRaySampler():
    def __init__(self):
        self._occ_gamma = None

    def update_occ_grid(self, occ_gamma):
        self._occ_gamma = occ_gamma

    def get_samples(self, rays, N_samples, perturb):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near = rays[:, -2:-1]
        far = rays[:, -1:]

        z_steps = torch.linspace(0, 1, N_samples // 2,
                                 device=rays.device)        # (N_samples)
        # z_steps = torch.logspace(-4, 0, N_samples, device=rays.device)       # (N_samples)
        z_vals = near * (1-z_steps) + far * z_steps
        z_vals = z_vals.expand(N_rays, N_samples // 2)

        if perturb > 0:  # perturb z_vals
            # (N_rays, N_samples-1) interval mid points
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
            perturb_rand = perturb * \
                torch.rand(z_vals.shape, device=rays.device)
            z_vals = lower + (upper - lower) * perturb_rand

        with torch.no_grad():
            # (N_rays, N_samples, 3)
            pts = rays_o.unsqueeze(
                1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
            point_logits = OccupancyGridModel.interpolate(self._occ_gamma, pts)
            point_probs = 1. / (1 + torch.exp(-point_logits))
            point_probs = 2 * (point_probs.clamp(min=0.5, max=1.0) - 0.5)

            # (N_rays, N_samples-1) interval mid points
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            z_vals_ = sample_pdf(
                z_vals_mid, point_probs[:, 1:-1], N_samples // 2, det=False).detach()
            # detach so that grad doesn't propogate to weights_coarse from here

            # sorting is important!
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        return z_vals
