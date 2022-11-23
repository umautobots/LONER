# ref: https://github.com/ashawkey/torch-ngp/blob/main/nerf/network_tcnn.py
import torch
import torch.nn as nn

import tinycudann as tcnn
import commentjson as json


class NeRF(nn.Module):
    def __init__(self, cfg, num_colors=3):
        super(NeRF, self).__init__()
        
        self.num_colors = num_colors

        with open(cfg) as config_file:
            config = json.load(config_file)

        self._model_sigma = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                         n_output_dims=16,
                                                         encoding_config=config["pos_encoding_sigma"],
                                                         network_config=config["sigma_network"])
        
        self._encoder_dir = tcnn.Encoding(n_input_dims=3, encoding_config=config["dir_encoding_intensity"])
        in_dim_intensity = self._encoder_dir.n_output_dims + 15
        self._model_intensity = tcnn.Network(n_input_dims=in_dim_intensity,
                                            n_output_dims=self.num_colors,
                                            network_config=config["intensity_network"])

    def forward(self, pos, dir, sigma_only=False):
        # x: [N, 3], scaled to [-1, 1]
        # d: [N, 3], normalized to [-1, 1]

        pos = (pos + 1) / 2
        h = self._model_sigma(pos)
        sigma = h[..., [0]]
        
        if sigma_only:
            return sigma 

        dir = (dir + 1) / 2
        dir = self._encoder_dir(dir)
        h_c = torch.cat([dir, h[..., 1:]], dim=-1)
        h_c = self._model_intensity(h_c)
        color = torch.sigmoid(h_c)
        
        return torch.cat([color, sigma], dim=-1)

    
class DecoupledNeRF(nn.Module):
    def __init__(self, cfg, num_colors=3):
        super(DecoupledNeRF, self).__init__()
        
        self._num_colors = num_colors

        with open(cfg) as config_file:
            config = json.load(config_file)
            
        pos_encoding_sigma = config["pos_encoding_sigma"]
        sigma_network = config["sigma_network"]
        pos_encoding_intensity = config["pos_encoding_intensity"]
        dir_encoding_intensity = config["dir_encoding_intensity"]
        intensity_network = config["intensity_network"]

        self._model_sigma = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                         n_output_dims=1,
                                                         encoding_config=pos_encoding_sigma,
                                                         network_config=sigma_network)
        
        self._pos_encoding = tcnn.Encoding(3, pos_encoding_intensity)
        self._dir_encoding = tcnn.Encoding(3, dir_encoding_intensity)
        network_in_dims = self._pos_encoding.n_output_dims + self._dir_encoding.n_output_dims
        self._model_intensity = tcnn.Network(n_input_dims=network_in_dims,
                                            n_output_dims=self._num_colors,
                                            network_config=intensity_network)

    def forward(self, pos, dir, sigma_only=False, detach_sigma=True):
        # x: [N, 3], scaled to [-1, 1]
        # d: [N, 3], normalized to [-1, 1]

        pos = (pos + 1) / 2
        if sigma_only:
            h = self._model_sigma(pos)
            sigma = h[..., [0]]
            return sigma
        elif detach_sigma:
            with torch.no_grad():
                h = self._model_sigma(pos)
                sigma = h[..., [0]]
        else:
            h = self._model_sigma(pos)
            sigma = h[..., [0]]
        
        dir = (dir + 1) / 2
        h_x = self._pos_encoding(pos)
        h_d = self._dir_encoding(dir)
        h_xd = torch.cat([h_x, h_d], dim=-1)
        h_c = self._model_intensity(h_xd)
        color = torch.sigmoid(h_c)
        
        return torch.cat([color, sigma], dim=-1)