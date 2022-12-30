import numpy as np
import torch
import torch.nn as nn


def img_to_mse(x, y):
    return torch.mean((x - y) ** 2)


def mse_to_psnr(x): 
    return -10. * torch.log(x) / \
        torch.log(torch.tensor([10.], device=x.device))


def to_uint8(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def get_weights_gt(sampled_depth, gt_depth, eps, norm=True):

    def _get_norm_pdf(x):
        return 1. / np.sqrt(2*np.pi) * torch.exp(-0.5*(x**2))

    def _get_norm_cdf(x):
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))

    sigma = eps / 3

    clip_a = (gt_depth - eps - gt_depth) / sigma
    clip_b = (gt_depth + eps - gt_depth) / sigma

    weights_gt = _get_norm_pdf((sampled_depth - gt_depth) / sigma) / \
        sigma / (_get_norm_cdf(clip_b) - _get_norm_cdf(clip_a))

    weights_gt_clipped = torch.heaviside(sampled_depth - (gt_depth-eps), torch.zeros_like(
        sampled_depth)) * torch.heaviside((gt_depth + eps) - sampled_depth, torch.zeros_like(sampled_depth)) * weights_gt

    if norm:
        weights_gt_clipped = weights_gt_clipped / \
            (weights_gt_clipped.sum(dim=1, keepdim=True) + 1e-6)
    return weights_gt_clipped


def get_logits_grad(z_vals, depth, eps=2, l_free=0.25, l_occ=2.5):
    def _func(x):
        _device = x.device
        y = l_free * torch.heaviside(-x-eps, torch.tensor([0.], device=_device)) \
            - l_occ * torch.heaviside(x+eps, torch.tensor([0.], device=_device)) * torch.heaviside(
                eps-x, torch.tensor([0.], device=_device))
        return y
    logits_grad = _func(z_vals-depth)
    return logits_grad
