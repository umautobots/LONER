from collections import defaultdict

import torch
import torch.nn as nn

from models.nerf_tcnn import DecoupledNeRF
from models.rendering_tcnn import render_rays, inference



# Holding module for all trainable variables
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        if cfg.model_type == 'nerf_decoupled':
            self.nerf_model = DecoupledNeRF(cfg.nerf_config, cfg.num_colors)
        else:
            raise NotImplementedError()

    def get_rgb_parameters(self):
        all_params =  list(self.nerf_model._model_intensity.parameters()) + \
               list(self.nerf_model._pos_encoding.parameters()) + \
               ([] if (self.nerf_model._dir_encoding is None) else list(self.nerf_model._dir_encoding.parameters()))
        return [p for p in all_params if p.requires_grad]

    def get_rgb_mlp_parameters(self):
        return list(self.nerf_model._model_intensity.parameters())

    def get_rgb_feature_parameters(self):
            params = list(self.nerf_model._pos_encoding.parameters()) + \
                   ([] if (self.nerf_model._dir_encoding is None) else list(self.nerf_model._dir_encoding.parameters()))
            return [p for p in params if p.requires_grad]

    def get_sigma_parameters(self):
        return [ p for p in list(self.nerf_model._model_sigma.parameters()) if p.requires_grad]
        
    def freeze_sigma_head(self, should_freeze=True):
        for p in self.get_sigma_parameters():
            p.requires_grad = not should_freeze

    def freeze_rgb_head(self, should_freeze=True):
        for p in self.get_rgb_parameters():
            p.requires_grad = not should_freeze
            
    def inference_points(self, xyz_, dir_, sigma_only):
        out = inference(self.nerf_model, xyz_, dir_, netchunk=0, sigma_only=sigma_only, meshing=True) # TODO: fix the bug when setting netchunk size 
        return out

    def forward(self, rays, ray_sampler, scale_factor, testing=False, camera=True, detach_sigma=True, return_variance=False):
        """Do batched inference on rays using chunk"""

        if testing:
            N_samples = self.cfg.render.N_samples_test
            perturb = 0.
        else:
            N_samples = self.cfg.render.N_samples_train
            perturb = self.cfg.render.perturb

        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.cfg.render.chunk):
            rays_chunk = rays[i:i+self.cfg.render.chunk, :]
            rendered_ray_chunks = \
                render_rays(rays_chunk,
                            ray_sampler,
                            self.nerf_model,
                            self.cfg.ray_range,
                            scale_factor,
                            N_samples=N_samples,
                            retraw=self.cfg.render.retraw,
                            perturb=perturb,
                            white_bkgd=self.cfg.render.white_bkgd,
                            raw_noise_std=self.cfg.render.raw_noise_std,
                            netchunk=self.cfg.render.netchunk,
                            num_colors=self.cfg.num_colors,
                            sigma_only=(not camera),
                            detach_sigma=detach_sigma,
                            return_variance=return_variance)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results


class OccupancyGridModel(nn.Module):
    def __init__(self, cfg):
        super(OccupancyGridModel, self).__init__()
        # 3D grid representing the logits (log-odds) of each voxel
        # log-odds = log(p/1-p)) where p is probability of voxel being occupied
        # a value of zero corresponds to equal likelihood of occupied and free

        self.cfg = cfg
        voxel_size = cfg.voxel_size
        self.occupancy_grid = nn.Parameter(torch.zeros(
            1, 1, voxel_size, voxel_size, voxel_size))

    def forward(self):
        return self.occupancy_grid

    @staticmethod
    def interpolate(occupancy_grid, ray_bin_centers, mode='bilinear'):
        # Uses torch grid_sample to compute the trilinear interpolation of occ_gamma to get values at ray_bin_centers
        # ray_bin_centers: (n_rays, n_bins, 3)
        n_rays, n_bins, _ = ray_bin_centers.shape
        grid_values = ray_bin_centers.reshape(1, 1, n_rays, n_bins, 3)
        bin_logits = nn.functional.grid_sample(
            occupancy_grid, grid_values, mode=mode, align_corners=False).reshape(n_rays, n_bins)
        return bin_logits
