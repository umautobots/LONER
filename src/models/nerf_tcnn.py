# ref: https://github.com/ashawkey/torch-ngp/blob/main/nerf/network_tcnn.py
import tinycudann as tcnn
import torch
import torch.nn as nn

class DecoupledNeRF(nn.Module):
    def __init__(self, cfg, num_colors=3):
        super(DecoupledNeRF, self).__init__()

        self._num_colors = num_colors

        self.cfg = cfg
        
        self._enable_view_dependence = cfg["enable_view_dependence"]

        pos_encoding_sigma = self.cfg["pos_encoding_sigma"]
        sigma_network = self.cfg["sigma_network"]
        pos_encoding_intensity = self.cfg["pos_encoding_intensity"]
        dir_encoding_intensity = self.cfg["dir_encoding_intensity"]
        intensity_network = self.cfg["intensity_network"]

        self._model_sigma = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                          n_output_dims=1,
                                                          encoding_config=pos_encoding_sigma,
                                                          network_config=sigma_network)
 
        self._pos_encoding = tcnn.Encoding(3, pos_encoding_intensity)

        if self._enable_view_dependence:
            self._dir_encoding = tcnn.Encoding(3, dir_encoding_intensity)
            network_in_dims = self._pos_encoding.n_output_dims + \
                self._dir_encoding.n_output_dims
        else:
            network_in_dims = self._pos_encoding.n_output_dims
            self._dir_encoding = None

        self._model_intensity = tcnn.Network(n_input_dims=network_in_dims,
                                             n_output_dims=self._num_colors,
                                             network_config=intensity_network)

        self._max_float = torch.finfo(self._model_intensity.dtype).max
        self._min_float = torch.finfo(self._model_intensity.dtype).min

        self._warn_infinite = True

    def forward(self, pos, dir, sigma_only=False, detach_sigma=True):
        # x: [N, 3], scaled to [-1, 1]
        # d: [N, 3], normalized to [-1, 1]

        pos = (pos + 1) / 2
        if sigma_only:
            h = self._model_sigma(pos)
            sigma = h[..., [0]]
        elif detach_sigma:
            with torch.no_grad():
                h = self._model_sigma(pos)
                sigma = h[..., [0]]
        else:
            h = self._model_sigma(pos)
            sigma = h[..., [0]]

        if not torch.isfinite(sigma).all():
            if self._warn_infinite:
                print("Warning: Clipping infinite outputs. Will not warn about this again (but it will happen again)")
                self._warn_infinite = False
            sigma = sigma.nan_to_num(posinf=self._max_float, neginf=self._min_float)
        
        if sigma_only:
            return sigma

        dir = (dir + 1) / 2
        h_x = self._pos_encoding(pos)

        if self._enable_view_dependence:
            h_d = self._dir_encoding(dir)
            h_xd = torch.cat([h_x, h_d], dim=-1)

            h_c = self._model_intensity(h_xd)
        else:
            h_c = self._model_intensity(h_x)
            
        color = torch.sigmoid(h_c)

        return torch.cat([color, sigma], dim=-1)
