import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.profiler
import torch.optim
import torchviz
import tqdm
from torch.profiler import ProfilerActivity, profile
# for visualization
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm

from common.pose_utils import WorldCube
from common.ray_utils import CameraRayDirections, rays_to_pcd, points_to_pcd, rays_to_o3d
import open3d as o3d
from common.settings import Settings
from mapping.keyframe import KeyFrame
from models.losses import (get_logits_grad, get_weights_gt, img_to_mse,
                           mse_to_psnr)
from models.model_tcnn import Model, OccupancyGridModel
from models.ray_sampling import UniformRaySampler, OccGridRaySampler

# Used for pre-creating random indices
MAX_POSSIBLE_LIDAR_RAYS = int(1e7)

ENABLE_WANDB = False

# Note: This is a bit of a deviation from how the rest of the code handles settings.
# But it makes sense here to be a bit extra explicit since we'll change these often

@dataclass
class OptimizationSettings:
    """ OptimizationSettings is a simple container for parameters for the optimizer
    """

    stage: int = 3
    num_iterations: int = 1
    freeze_poses: bool = False  # Fix the map, only optimize poses
    latest_kf_only: bool = False # If freeze poses is false but this is true, only the most recent pose is optimized
    freeze_sigma_mlp: bool = False
    freeze_rgb_mlp: bool = False


class Optimizer:
    """ The Optimizer module is used to run iterations of the Loner Optimization.

    The KeyFrameManager supplies the Optimizer with a window of KeyFrame objects,
    which the Optimizer then uses to draw samples and iterate the optimization
    """

    ## Constructor
    # @param settings: Optimizer-specific settings. See the example settings for details.
    # @param calibration: Calibration-related settings. See the example settings for details
    # @param world_cube: The world cube pre-computed that is used to scale the world.
    # @param device: Which device to put the data on and run the optimizer on
    def __init__(self, settings: Settings, calibration: Settings, world_cube: WorldCube, device: int,
                 use_gt_poses: bool = False, lidar_only: bool = True,
                 enable_sky_segmentation: bool = True):

        self._settings = settings
        self._calibration = calibration
        self._device = device
        
        self._use_gt_poses = use_gt_poses

        self._optimization_settings = OptimizationSettings()

        self._lidar_only = lidar_only

        self._model_config = settings.model_config

        self._scale_factor = world_cube.scale_factor

        self._data_prep_device = 'cpu' if settings.data_prep_on_cpu else self._device

        self._world_cube = world_cube.to(self._data_prep_device)

        self._ray_range = torch.Tensor(
            self._model_config.model.ray_range).to(self._data_prep_device)

        # Main Model
        self._model = Model(self._model_config.model)

        if self._settings.samples_selection.strategy == 'OGM':
            # Occupancy grid
            self._occupancy_grid_model = OccupancyGridModel(
                self._model_config.model.occ_model).to(self._device)

            self._occupancy_grid = self._occupancy_grid_model()

            occ_grid_parameters = [
                p for p in self._occupancy_grid_model.parameters() if p.requires_grad]
            self._occupancy_grid_optimizer = torch.optim.SGD(
                occ_grid_parameters, lr=self._model_config.model.occ_model.lr)

            self._ray_sampler = OccGridRaySampler()
            self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())
        elif self._settings.samples_selection.strategy == 'UNIFORM':
            self._ray_sampler = UniformRaySampler()
        else:
            raise RuntimeError(
                f"Can't find samples_selection strategy: {self._settings.samples_selection.strategy}")

        if not self._lidar_only:
            # We pre-create random numbers to lookup at runtime to save runtime.
            # This kills a lot of memory, but saves a ton of runtime
            # self._lidar_shuffled_indices = torch.randperm(MAX_POSSIBLE_LIDAR_RAYS)
            self._rgb_shuffled_indices = torch.randperm(
                calibration.camera_intrinsic.width * calibration.camera_intrinsic.height)

            self._cam_ray_directions = CameraRayDirections(calibration, device=self._data_prep_device)
    
        self._keyframe_count = 0
        self._global_step = 0


        self._keyframe_schedule = self._settings["keyframe_schedule"]

        self._optimizer = None

        self._num_rgb_samples = self._settings.num_samples.rgb
        self._num_lidar_samples = self._settings.num_samples.lidar

        self._enable_sky_segmentation = enable_sky_segmentation

    ## Run one or more iterations of the optimizer, as specified by the stored settings
    # @param keyframe_window: The set of keyframes to use in the optimization.
    def iterate_optimizer(self, keyframe_window: List[KeyFrame], optimizer_settings: OptimizationSettings = None) -> float:

        # Look at the keyframe schedule and figure out which iteration schedule to use
        cumulative_kf_idx = 0
        for item in self._keyframe_schedule:
            kf_count = item["num_keyframes"]
            iteration_schedule = item["iteration_schedule"]

            cumulative_kf_idx += kf_count
            if cumulative_kf_idx >= self._keyframe_count + 1 or kf_count == -1:
                break

        num_its = sum(i["num_iterations"] for i in iteration_schedule)

        if self._settings.debug.profile_optimizer:
            prof_dir = f"{self._settings.log_directory}/profile"
            os.makedirs(prof_dir, exist_ok=True)

            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         profile_memory=True,
                         record_shapes=True,
                         with_stack=True,
                         with_modules=True,
                         schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_its - 2),
                         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{prof_dir}/tensorboard_optimizer/"))
            
            prof.start()
            start_time = time.time()
            result = self._do_iterate_optimizer(keyframe_window, iteration_schedule, prof, optimizer_settings=optimizer_settings)
            end_time = time.time()
            prof.stop()

            print(f"Elapsed Time: {end_time - start_time}. Per Iteration: {(end_time - start_time)/num_its}, Its/Sec: {1/((end_time - start_time)/num_its)}")

        else:          
            start_time = time.time()
            result = self._do_iterate_optimizer(keyframe_window, iteration_schedule, optimizer_settings=optimizer_settings)
            end_time = time.time()
            elapsed_time = end_time - start_time

            log_file = f"{self._settings.log_directory}/timing.csv"
            with open(log_file, 'a+') as f:
                f.write(f"{num_its},{elapsed_time}\n")
            print(f"Elapsed Time: {end_time - start_time}. Per Iteration: {(end_time - start_time)/num_its}, Its/Sec: {1/((end_time - start_time)/num_its)}")
       
        self._keyframe_count += 1
        return result

    def _do_iterate_optimizer(self, keyframe_window: List[KeyFrame], iteration_schedule: dict, 
                              profiler: profile = None, optimizer_settings: OptimizationSettings = None) -> float:
        
        if len(keyframe_window) == 1:
            keyframe_window[0].is_anchored = True

        if len(iteration_schedule) > 1 and self._settings.skip_pose_refinement:
            iteration_schedule = iteration_schedule[1:]

        # For each iteration config, have a list of the losses
        losses_log = []
        depth_eps_log = []

        for iteration_config in iteration_schedule:
            losses_log.append([])
            depth_eps_log.append([])
            
            if optimizer_settings is None:
                self._optimization_settings.freeze_poses = iteration_config["fix_poses"] or self._settings.fix_poses or self._use_gt_poses

                
                if "latest_kf_only" in iteration_config:
                    self._optimization_settings.latest_kf_only = iteration_config["latest_kf_only"]
                else:
                    self._optimization_settings.latest_kf_only = False

                self._optimization_settings.freeze_rgb_mlp = iteration_config["fix_rgb_mlp"]
                self._optimization_settings.freeze_sigma_mlp = iteration_config["fix_sigma_mlp"]
                self._optimization_settings.num_iterations = iteration_config["num_iterations"]
                self._optimization_settings.stage = 1 if self._lidar_only else iteration_config["stage"]
            else:
                self._optimization_settings = optimizer_settings

            self._optimization_settings.freeze_poses = self._optimization_settings.freeze_poses or self._settings.fix_poses or self._use_gt_poses

            if self._settings.samples_selection.strategy == 'OGM':
                self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())

            self._model.freeze_sigma_head(self._optimization_settings.freeze_sigma_mlp)
            self._model.freeze_rgb_head(self._optimization_settings.freeze_rgb_mlp)

            optimize_poses = not self._optimization_settings.freeze_poses

            if self._optimization_settings.latest_kf_only:
                most_recent_ts = -1
                for kf in keyframe_window:
                    if kf.get_time() > most_recent_ts:
                        most_recent_ts = kf.get_time()
                        most_recent_kf = kf

                active_keyframe_window = [most_recent_kf]
            else:
                active_keyframe_window = keyframe_window

            for kf in active_keyframe_window:
                if not kf.is_anchored:
                    kf.get_lidar_pose().set_fixed(not optimize_poses)
            
            tracking = (not self._optimization_settings.freeze_poses) and \
                self._optimization_settings.freeze_rgb_mlp and self._optimization_settings.freeze_sigma_mlp
            
            if tracking:
                optimizable_poses = [kf.get_lidar_pose().get_pose_tensor() for kf in active_keyframe_window if not kf.is_anchored]
                self._optimizer = torch.optim.Adam([{'params': optimizable_poses, 'lr': self._model_config.train.lrate_pose}])
            elif optimize_poses:
                optimizable_poses = [kf.get_lidar_pose().get_pose_tensor() for kf in active_keyframe_window if not kf.is_anchored]
                        
                self._optimizer = torch.optim.Adam([{'params': self._model.get_sigma_parameters(), 'lr': self._model_config.train.lrate_sigma_mlp},
                                                    {'params': self._model.get_rgb_mlp_parameters(), 'lr': self._model_config.train.lrate_rgb,
                                                        'weight_decay': self._model_config.train.rgb_weight_decay},
                                                    {'params': self._model.get_rgb_feature_parameters(), 'lr': self._model_config.train.lrate_rgb},
                                                    {'params': optimizable_poses, 'lr': self._model_config.train.lrate_pose}])

            else:
                self._optimizer = torch.optim.Adam([{'params': self._model.get_sigma_parameters(), 'lr': self._model_config.train.lrate_sigma_mlp},
                                                    {'params': self._model.get_rgb_mlp_parameters(), 'lr': self._model_config.train.lrate_rgb,
                                                        'weight_decay': self._model_config.train.rgb_weight_decay},
                                                    {'params': self._model.get_rgb_feature_parameters(), 'lr': self._model_config.train.lrate_rgb}])
                
            lrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._model_config.train.lrate_gamma)
            
            for it_idx in tqdm.tqdm(range(self._optimization_settings.num_iterations)):
                lidar_rays, lidar_depths = None, None
                camera_rays, camera_intensities = None, None
                
                # Bookkeeping for occ update
                self._results_lidar = None
        
                camera_samples, lidar_samples = None, None

                for kf_idx, kf in enumerate(active_keyframe_window):                
                    if self.should_enable_lidar():
                        if self._settings.rays_selection.strategy == 'RANDOM':
                            lidar_indices = torch.randint(len(kf.get_lidar_scan()), (self._num_lidar_samples,))
                        elif self._settings.rays_selection.strategy == 'MASK':
                            mask_index_map = kf.get_lidar_scan().mask.nonzero(as_tuple=True)[0]
                            mask_indices = torch.randint(len(mask_index_map), (self._num_lidar_samples,))
                            lidar_indices = mask_index_map[mask_indices]
                        elif self._settings.rays_selection.strategy == 'FIXED':
                            lidar_indices = torch.arange(self._num_lidar_samples)
                        else:
                            raise RuntimeError(
                                f"Can't find rays_selection strategy: {self._settings.rays_selection.strategy}")

                        sky_dirs = kf.get_lidar_scan().sky_rays
                        if self._settings.num_samples.sky > 0 and self._enable_sky_segmentation and sky_dirs.nelement() > 0:
                            sky_indices = torch.randint(0, sky_dirs.shape[1], (self._settings.num_samples.sky,))
                        else:
                            sky_indices = None

                        new_rays, new_depths = kf.build_lidar_rays(lidar_indices, self._ray_range, self._world_cube, self._use_gt_poses, sky_indices=sky_indices)

                        if self._settings.debug.write_ray_point_clouds and self._global_step % 100 == 0:
                            os.makedirs(f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_rays", exist_ok=True)
                            os.makedirs(f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_origins", exist_ok=True)
                            rays_fname = f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_rays/rays_{self._global_step}_{it_idx}.pcd"
                            origins_fname = f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_origins/origins_{self._global_step}_{it_idx}.pcd"
                            rays_to_pcd(new_rays, new_depths, rays_fname, origins_fname)

                        if self._settings.debug.store_ray:
                            sky_mask = torch.zeros_like(new_depths)
                            current_mask = torch.zeros_like(new_depths)
                            if sky_indices != None:
                                sky_mask[-self._settings.num_samples.sky:] = 1
                            if kf_idx == len(active_keyframe_window)-1:
                                current_mask = torch.ones_like(new_depths)

                            if it_idx==0 and kf_idx==0:
                                acc_rays = new_rays
                                acc_depths = new_depths
                                acc_sky_mask = sky_mask
                                acc_current_mask = current_mask
                            else:
                                acc_rays = torch.vstack((acc_rays, new_rays))
                                acc_depths = torch.hstack((acc_depths, new_depths))
                                acc_sky_mask = torch.hstack((acc_sky_mask, sky_mask))
                                acc_current_mask = torch.hstack((acc_current_mask, current_mask))

                        if lidar_rays is None:
                            lidar_rays = new_rays
                            lidar_depths = new_depths
                        else:
                            lidar_rays = torch.vstack((lidar_rays, new_rays))
                            lidar_depths = torch.cat((lidar_depths, new_depths))
                        
                        lidar_samples = (lidar_rays.to(self._device).float(), lidar_depths.to(self._device).float())

                    if self.should_enable_camera():

                        # Get all the uniform samples first
                        start_idx = torch.randint(len(self._rgb_shuffled_indices), (1,))

                        im_end_idx = start_idx[0] + self._num_rgb_samples
                        im_indices = self._rgb_shuffled_indices[start_idx[0]:im_end_idx]
                        im_indices = im_indices % len(self._rgb_shuffled_indices)
                        
                        new_cam_rays, new_cam_intensities = kf.build_camera_rays(
                            im_indices, self._ray_range,
                            self._cam_ray_directions, self._world_cube,
                            self._use_gt_poses,
                            self._settings.detach_rgb_from_poses)

                        if camera_rays is None:
                            camera_rays, camera_intensities = new_cam_rays, new_cam_intensities
                        else:
                            camera_rays = torch.vstack((camera_rays, new_cam_rays))
                            camera_intensities = torch.vstack((camera_intensities, new_cam_intensities))

                        if self._settings.debug.write_ray_point_clouds:
                            os.makedirs(f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_rays", exist_ok=True)
                            os.makedirs(f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_origins", exist_ok=True)
                            rays_fname = f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_rays/rays_{self._global_step}_{it_idx}.pcd"
                            origins_fname = f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_origins/origins_{self._global_step}_{it_idx}.pcd"
                            rays_to_pcd(new_cam_rays, torch.ones_like(new_cam_rays[:,0]) * .1, rays_fname, origins_fname, new_cam_intensities)
                    
                        camera_samples = (camera_rays.to(self._device).float(), camera_intensities.to(self._device).float())
                
                if self._settings.debug.store_ray:
                    loss, std, js = self.compute_loss(camera_samples, lidar_samples, it_idx, tracking=tracking)
                    if it_idx==0:
                        acc_std = std
                        acc_js = js
                    else:
                        acc_std = torch.hstack((acc_std, std))
                        acc_js = torch.hstack((acc_js, js))
                else:
                    loss = self.compute_loss(camera_samples, lidar_samples, it_idx, tracking=tracking)

                losses_log[-1].append(loss.detach().cpu().item())

                if self.should_enable_lidar():
                    depth_eps_log[-1].append(self._depth_eps)

                if self._settings.debug.draw_comp_graph:
                    graph_dir = f"{self._settings.log_directory}/graphs"
                    os.makedirs(graph_dir, exist_ok=True)
                    loss_dot = torchviz.make_dot(loss)
                    loss_dot.format = "png"
                    loss_dot.render(directory=graph_dir, filename=f"iteration_{self._global_step}")

                loss.backward(retain_graph=False)

                for kf in keyframe_window:          
                    if kf.get_lidar_pose().get_pose_tensor().grad is not None and not kf.get_lidar_pose().get_pose_tensor().grad.isfinite().all():
                        raise RuntimeError("Fatal: Encountered invalid gradient in pose.")

                for kf in keyframe_window:
                    if not kf.get_lidar_pose().get_pose_tensor().isfinite().all():
                        raise RuntimeError("Fatal: Encountered invalid pose tensor.")
                
                self._optimizer.step()

                lrate_scheduler.step()

                self._optimizer.zero_grad(set_to_none=True)

                if self.should_enable_lidar() and self._settings.samples_selection.strategy == 'OGM' and \
                        self._global_step % self._model_config.model.occ_model.N_iters_acc == 0:
                    self._step_occupancy_grid()
                self._global_step += 1

                if profiler is not None:
                    profiler.step()

            if self._settings.debug.store_ray:
                os.makedirs(f"{self._settings['log_directory']}/rays/lidar", exist_ok=True)
                rays_fname = f"{self._settings['log_directory']}/rays/lidar/kf_{self._keyframe_count}.pcd"
                os.makedirs(f"{self._settings['log_directory']}/rays/sky_mask", exist_ok=True)
                sky_mask_fname = f"{self._settings['log_directory']}/rays/sky_mask/kf_{self._keyframe_count}.pt"
                os.makedirs(f"{self._settings['log_directory']}/rays/curr_mask", exist_ok=True)
                curr_mask_fname = f"{self._settings['log_directory']}/rays/curr_mask/kf_{self._keyframe_count}.pt"
                os.makedirs(f"{self._settings['log_directory']}/rays/std", exist_ok=True)
                std_fname = f"{self._settings['log_directory']}/rays/std/kf_{self._keyframe_count}.pt"
                os.makedirs(f"{self._settings['log_directory']}/rays/js", exist_ok=True)
                js_fname = f"{self._settings['log_directory']}/rays/js/kf_{self._keyframe_count}.pt"

                o3d_pc = rays_to_o3d(acc_rays, acc_depths, self._world_cube)
                o3d.io.write_point_cloud(rays_fname, o3d_pc, print_progress=True)
                torch.save(acc_sky_mask, sky_mask_fname)
                torch.save(acc_current_mask, curr_mask_fname)
                torch.save(acc_std, std_fname)
                torch.save(acc_js, js_fname)

        if self._settings.debug.log_losses:
            graph_dir = f"{self._settings.log_directory}/losses/keyframe_{self._keyframe_count}"
            os.makedirs(graph_dir, exist_ok=True)
            for log_idx, log in enumerate(losses_log):
                with open(f"{graph_dir}/phase_{log_idx}.csv", 'w+') as log_file:
                    log = [str(l) for l in log]
                    log_file.write("\n".join(log))
            graph_dir = f"{self._settings.log_directory}/depth_eps/keyframe_{self._keyframe_count}"
            os.makedirs(graph_dir, exist_ok=True)
            for log_idx, log in enumerate(depth_eps_log):
                with open(f"{graph_dir}/phase_{log_idx}.csv", 'w+') as log_file:
                    log = [str(l) for l in log]
                    log_file.write("\n".join(log))

    ## @returns whether or not the lidar should be used, as indicated by the settings
    def should_enable_lidar(self) -> bool:
        return self._optimization_settings.stage in [1, 3] \
                    and (not self._optimization_settings.freeze_sigma_mlp \
                         or not self._optimization_settings.freeze_poses)

    ## @returns whether or not the camera should be use, as indicated by the settings
    def should_enable_camera(self) -> bool:
        if self._lidar_only:
            return False

        return self._optimization_settings.stage in [2, 3] \
                and (not self._optimization_settings.freeze_rgb_mlp \
                     or (not self._settings.detach_rgb_from_poses \
                         and not self._optimization_settings.freeze_poses))

    ## For the given camera and lidar rays, compute and return the differentiable loss
    def compute_loss(self, camera_samples: Tuple[torch.Tensor, torch.Tensor], 
                           lidar_samples: Tuple[torch.Tensor, torch.Tensor],
                           iteration_idx: int,
                           override_enables: bool = False,
                           tracking=False) -> torch.Tensor:
        scale_factor = self._scale_factor.to(self._device).float()
        
        loss = 0
        wandb_logs = {}

        if (override_enables or self.should_enable_lidar()) and lidar_samples is not None:

            if self._model_config.loss.decay_los_lambda:
                los_lambda = max(self._model_config.loss.los_lambda * (self._model_config.loss.los_lambda_decay_rate ** (
                    (self._global_step + 1) / (self._model_config.loss.los_lambda_decay_steps))), self._model_config.loss.min_los_lambda)
            else:
                los_lambda = self._model_config.loss.los_lambda
            
            # rays = [origin, direction, viewdir, <ignore>, near limit, far limit]
            lidar_rays, lidar_depths = lidar_samples

            lidar_rays = lidar_rays.reshape(-1, lidar_rays.shape[-1])
            lidar_depths = lidar_depths.reshape(-1, 1)

            far = lidar_rays[:,-1]
            transparent_rays = (lidar_depths.view(-1,1) > far)[...,0]
            
            opaque_rays = torch.logical_and((lidar_depths > 0)[..., 0], ~transparent_rays)

            # Rendering lidar rays. Results need to be in class for occ update to happen
            self._results_lidar = self._model(lidar_rays, self._ray_sampler, scale_factor, camera=False, return_variance=True)

            # (N_rays, N_samples)
            # Depths along ray
            self._lidar_depth_samples_fine = self._results_lidar['samples_fine'] * scale_factor

            self._lidar_depths_gt = lidar_depths * scale_factor
            weights_pred_lidar = self._results_lidar['weights_fine']

            variance = self._results_lidar["variance"]
            
            # Compute JS divergence (also calculate when using fix depth_eps for visualization)
            mean = torch.sum(self._lidar_depth_samples_fine * weights_pred_lidar, axis=1) / (torch.sum(weights_pred_lidar, axis=1) + 1e-10) # weighted mean # [N_rays]
            var = torch.sum((self._lidar_depth_samples_fine-torch.unsqueeze(mean, dim=-1))**2 * weights_pred_lidar, axis=1) / (torch.sum(weights_pred_lidar, axis=1) + 1e-10) + 1e-10 # [N_rays]
            std = torch.sqrt(var) # [N_rays]
            std_out = std
            mean, var, std = torch.unsqueeze(mean, 1), torch.unsqueeze(var, 1), torch.unsqueeze(std, 1)
            eps_min = self._model_config.loss.min_depth_eps
            js_score = self.calculate_JS_divergence(self._lidar_depths_gt, eps_min/3., mean, std).squeeze()
            js_score_out = js_score

            # add depth loss
            depth_euc_fine = self._results_lidar['depth_fine'].unsqueeze(
                1) * scale_factor
            depth_loss_fine = nn.functional.mse_loss(
                depth_euc_fine[opaque_rays, 0], self._lidar_depths_gt[opaque_rays, 0])

            loss += self._model_config.loss.depthloss_lambda * depth_loss_fine
            wandb_logs['loss_depth'] = depth_loss_fine.item()

            loss_selection = self._model_config.loss.loss_selection
            
            if loss_selection == 'L1_JS' or loss_selection == 'L2_JS':
                min_js_score = self._model_config.loss.JS_loss.min_js_score
                max_js_score = self._model_config.loss.JS_loss.max_js_score
                alpha = self._model_config.loss.JS_loss.alpha
                js_score[js_score<min_js_score] = 0
                js_score[js_score>max_js_score] = max_js_score
                eps_dynamic = eps_min*(1+(alpha * js_score))
                eps_dynamic = torch.unsqueeze(eps_dynamic, dim=-1).detach()
                self._depth_eps = float(np.average(eps_dynamic.detach().cpu().numpy()))
                weights_gt_lidar = get_weights_gt(self._lidar_depth_samples_fine, self._lidar_depths_gt, eps=eps_dynamic) # [N_rays, N_samples]

                weights_gt_lidar[~opaque_rays, :] = 0

                if self._settings.debug.visualize_loss and self._keyframe_count >= 0:
                    # viz_idx = np.where(js_score.detach().cpu().numpy() == max_js_score)[0] # show rays that haven't converged
                    viz_idx = np.array([0]) # show the first ray
                    self.visualize_loss(iteration_idx, viz_idx, opaque_rays, weights_gt_lidar, weights_pred_lidar, \
                                mean, var, js_score, \
                                self._lidar_depth_samples_fine, self._lidar_depths_gt, eps_dynamic, \
                                depth_euc_fine)

            elif loss_selection == 'L1_LOS' or loss_selection == 'L2_LOS':
                if self._model_config.loss.decay_depth_eps:
                    self._depth_eps = max(self._model_config.loss.depth_eps * (self._model_config.loss.depth_eps_decay_rate ** (
                                    iteration_idx / (self._model_config.loss.depth_eps_decay_steps))), self._model_config.loss.min_depth_eps)
                else:
                    self._depth_eps = self._model_config.loss.depth_eps
                weights_gt_lidar = get_weights_gt(
                    self._lidar_depth_samples_fine, self._lidar_depths_gt, eps=self._depth_eps)
                
                weights_gt_lidar[~opaque_rays, :] = 0
                
                if self._settings.debug.visualize_loss and self._keyframe_count >= 0:
                    # viz_idx = np.where(js_score.detach().cpu().numpy() == max_js_score)[0] # show rays that haven't converged
                    viz_idx = np.array([0]) # show the first ray
                    self.visualize_loss(iteration_idx, viz_idx, opaque_rays, weights_gt_lidar, weights_pred_lidar, \
                                    mean, var, js_score, \
                                    self._lidar_depth_samples_fine, self._lidar_depths_gt, self._depth_eps, \
                                    depth_euc_fine)

            elif loss_selection == 'KL':
                # loss = self._model.kl_loss(lidar_rays, lidar_depths, self._ray_sampler, scale_factor, camera=False, return_variance=True).sum()
                # target_distribution = norm.pdf(self._lidar_depth_samples_fine.detach().cpu().numpy(), self._lidar_depths_gt.detach().cpu().numpy(), eps_min/3.)
                # target_distribution = torch.tensor(target_distribution).to(self._device)
                # loss = self.calculate_KL_divergence_from_raw_distribution(target_distribution[:,:-1]/torch.sum(target_distribution[:,:-1], 1), weights_pred_lidar[:,:-1]/torch.sum(weights_pred_lidar[:,:-1], 1))
                loss = self.calculate_KL_divergence(self._lidar_depths_gt, eps_min/3., mean, std).squeeze().mean()
                depth_loss_los_fine = torch.tensor(0)
                self._depth_eps = 0
            else:
                raise ValueError(f"Can't use unknown Loss {loss_selection}")

            if self._settings.debug.draw_samples:
                points = self._results_lidar["points_fine"].view(-1, 3).detach().cpu()
                weights = self._results_lidar["weights_fine"].view(-1, 1).detach().cpu().flatten()

                points *= self._world_cube.scale_factor
                points -= self._world_cube.shift

                points_est = points[weights > 1e-5]
                weights_est = weights[weights > 1e-5]

                weights_gt = weights_gt_lidar.view(-1, 1).detach().cpu().flatten()
                points_gt = points[weights_gt > 1e-5]
                weights_gt = weights_gt[weights_gt > 1e-5]

                samples_dir = f"{self._settings.log_directory}/samples"
                os.makedirs(samples_dir, exist_ok=True)
                points_to_pcd(points_est, f"{samples_dir}/samples_kf{self._keyframe_count}_it{iteration_idx}.pcd", weights_est)
                points_to_pcd(points_gt, f"{samples_dir}/samples_kf{self._keyframe_count}_it{iteration_idx}_gt.pcd", weights_gt)

            if self._settings.debug.draw_rays_eps:
                rays_eps_dir = f"{self._settings.log_directory}/rays_eps"
                os.makedirs(rays_eps_dir, exist_ok=True)
                rays_fname = f"{rays_eps_dir}/rays_kf{self._keyframe_count}_it{iteration_idx}.pcd"
                origins_fname = f"{rays_eps_dir}/origins_kf{self._keyframe_count}_it{iteration_idx}.pcd"

                lidar_rays, lidar_depths = lidar_samples
                eps_dynamic_max = eps_min*(1+(alpha * self._model_config.loss.JS_loss.max_js_score)) + 1e-5
                eps_dynamic = eps_dynamic / eps_dynamic_max
                color = eps_dynamic.repeat(1, 3)
                rays_to_pcd(lidar_rays, lidar_depths, rays_fname, origins_fname, color)

            # add LOS-based loss
            if loss_selection == 'L1_JS' or loss_selection == 'L1_LOS':                
                depth_loss_los_fine = nn.functional.l1_loss(
                    weights_pred_lidar, weights_gt_lidar)
            elif loss_selection == 'L2_JS' or loss_selection == 'L2_LOS':                
                depth_loss_los_fine = nn.functional.mse_loss(
                    weights_pred_lidar, weights_gt_lidar)
            loss += los_lambda * depth_loss_los_fine
            wandb_logs['loss_lidar_los'] = depth_loss_los_fine.item()

            # add opacity loss
            loss_opacity_lidar = torch.abs(
                self._results_lidar['opacity_fine'][opaque_rays] - 1).mean()

            loss += loss_opacity_lidar
            wandb_logs['loss_opacity_lidar'] = loss_opacity_lidar.item()

            n_depth = lidar_depths.shape[0]

            depth_peaks_lidar = self._lidar_depth_samples_fine[torch.arange(
                n_depth), weights_pred_lidar.argmax(dim=1)].detach()
            loss_unimod_lidar = nn.functional.mse_loss(
                depth_peaks_lidar, depth_euc_fine[..., 0])
            wandb_logs['loss_unimod_lidar'] = loss_unimod_lidar.item()

            if self._model_config.loss.decay_los_lambda:
                los_lambda = max(self._model_config.loss.los_lambda * (self._model_config.loss.los_lambda_decay_rate ** (
                    (self._global_step + 1) / (self._model_config.loss.los_lambda_decay_steps))), self._model_config.loss.min_los_lambda)

            wandb_logs['los_lambda'] = los_lambda
            if loss_selection != 'KL':
                wandb_logs['depth_eps'] = self._depth_eps

        if (override_enables or self.should_enable_camera()) and camera_samples is not None:
            # camera_samples is organized as [cam_rays, cam_intensities]
            # cam_rays = [origin, direction, viewdir, ray_i_grid, ray_j_grid, near limit, far limit]
            cam_rays, cam_intensities = camera_samples

            cam_intensities = cam_intensities.detach()

            cam_rays = cam_rays.reshape(-1, cam_rays.shape[-1])
            cam_intensities = cam_intensities.reshape(
                -1, self._model_config.model.num_colors)

            results_cam = self._model(
                cam_rays, self._ray_sampler, scale_factor, detach_sigma=self._settings.detach_rgb_from_sigma)

            psnr_fine = mse_to_psnr(
                img_to_mse(results_cam['rgb_fine'], cam_intensities))

            wandb_logs['psnr_fine'] = psnr_fine

            cam_loss_fine = nn.functional.l1_loss(
                results_cam['rgb_fine'], cam_intensities)
            loss += self._model_config.loss.cam_lambda * cam_loss_fine
            wandb_logs['loss_cam_fine'] = cam_loss_fine.item()

            loss_opacity_cam = torch.abs(
                results_cam['opacity_fine'] - 1).mean()
            loss += loss_opacity_cam
            wandb_logs['loss_opacity_cam'] = loss_opacity_cam.item()

            cam_samples_fine = results_cam['samples_fine'] * scale_factor
            weights_pred_cam = results_cam['weights_fine']

            n_color = cam_intensities.shape[0]
            depths_peaks_cam = cam_samples_fine[torch.arange(
                n_color), weights_pred_cam.argmax(dim=1)].detach()

            loss_unimod_cam = nn.functional.mse_loss(
                depths_peaks_cam, results_cam['depth_fine'] * scale_factor)
            wandb_logs['loss_unimod_cam'] = loss_unimod_cam.item()

            depths_weighted_mean = (cam_samples_fine * weights_pred_cam).sum(
                dim=1, keepdim=True)/(weights_pred_cam.sum(dim=1, keepdim=True) + 1e-6)
            depths_weighted_var = (weights_pred_cam * (cam_samples_fine - depths_weighted_mean)**2).sum(
                dim=1, keepdim=True)/(weights_pred_cam.sum(dim=1, keepdim=True) + 1e-6)

            loss_std_cam = torch.abs(torch.sqrt(
                depths_weighted_var + 1e-8).mean() - (1 + 1e-8))
            loss += self._model_config.loss.std_lambda * loss_std_cam
            wandb_logs['loss_std_cam'] = loss_std_cam.item()

        if isinstance(loss, int) and loss == 0:
            print("Warning: zero loss")
            
        assert not torch.isnan(loss), "NaN Loss Encountered"
        
        if self._settings.debug.store_ray:
            return loss, std_out.detach(), js_score_out.detach()
        else:
            return loss

    # @precond: This MUST be called after compute_loss!!
    def _step_occupancy_grid(self):
        lidar_points = self._results_lidar['points_fine'].detach()
        point_logits = OccupancyGridModel.interpolate(
            self._occupancy_grid, lidar_points)
        point_logits_grad = get_logits_grad(
            self._lidar_depth_samples_fine, self._lidar_depths_gt)
        point_logits.backward(
            gradient=point_logits_grad, retain_graph=True)
        self._occupancy_grid_optimizer.step()
        self._occupancy_grid = self._occupancy_grid_model()
        self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())
        self._occupancy_grid_optimizer.zero_grad()

    def calculate_KL_divergence_from_raw_distribution(self, p, q):
        return torch.sum(torch.where(p != 0, p * torch.log(p / (q + 1e-10)), 0.))

    def calculate_KL_divergence(self, mean1, std1, mean2, std2):
        var1 = std1 * std1
        var2 = std2 * std2
        a = torch.log(std2/std1) 
        num = var1 + (mean1 - mean2)**2
        den = 2 * var2
        b = num / den
        return a + b - 0.5
    
    def calculate_JS_divergence(self, mean1, std1, mean2, std2):
        mean_m = 0.5 * (mean1+mean2)
        std_m = 0.5 * torch.sqrt(std1**2+std2**2)
        return 0.5 * self.calculate_KL_divergence(mean1, std1, mean_m, std_m) + 0.5 * self.calculate_KL_divergence(mean2, std2, mean_m, std_m)

    def visualize_loss(self, iteration_idx, viz_idx: np.ndarray, opaque_rays: torch.Tensor, weights_gt_lidar: torch.Tensor, weights_pred_lidar: torch.Tensor,
                        mean: torch.Tensor, var: torch.Tensor, js_score: torch.Tensor, \
                        s_vals_lidar: torch.Tensor, depth_gt_lidar: torch.Tensor, eps_: torch.Tensor, \
                        expected_depth: torch.Tensor)->None:
        
        opaque_rays = opaque_rays.detach().cpu().numpy()
        weights_gt_lidar = weights_gt_lidar.detach().cpu().numpy()
        weights_pred_lidar = weights_pred_lidar.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy()
        var = var.detach().cpu().numpy()
        js_score = js_score.detach().cpu().numpy()
        s_vals_lidar = s_vals_lidar.detach().cpu().numpy()
        depth_gt_lidar = depth_gt_lidar.detach().cpu().numpy()
        expected_depth = expected_depth.detach().cpu().numpy()

        if isinstance(eps_, type(torch)):
            eps_ = eps_.detach().cpu().numpy()
        if iteration_idx > 0:
            # max_js_ids = np.where(js_score == self._model_config.loss.JS_loss.max_js_score)[0]
            # opaque_ids = np.where(opaque_rays == True)[0]
            # print('maxjs_count: ', len(np.intersect1d(max_js_ids, opaque_ids)) )
            # return
            
            use_js = (self._model_config.loss.loss_selection == 'L1_JS' or self._model_config.loss.loss_selection == 'L2_JS')
            depth_only = (self._model_config.loss.depthloss_lambda!=0 and self._model_config.loss.los_lambda==0)

            for j in viz_idx:
                if not opaque_rays[j]:
                    continue
                print('ray idx:', j)
                x = s_vals_lidar[j]
                y = weights_pred_lidar[j]
                y_gt = weights_gt_lidar[j]
                u = mean[j]
                variance = var[j]
                print("sample mean: ", u, " var: ", variance)
                plt.figure(figsize=(15, 10))
                plt.tight_layout()

                font = {'family' : 'normal',
                        'weight' : 'normal',
                        'size'   : 30} # 18
                matplotlib.rc('font', **font)

                depth_gt_lidar = np.array(np.squeeze(depth_gt_lidar)).reshape((-1))
                expected_depth = np.array(np.squeeze(expected_depth)).reshape((-1))
                eps_min = self._model_config.loss.min_depth_eps
                if use_js:
                    eps_dynamic_ = eps_.cpu()
                    print("u_gt:", depth_gt_lidar[j], "eps: ", eps_dynamic_[j])
                else:
                    depth_eps_ = eps_
                    print("u_gt:", depth_gt_lidar[j], "eps: ", depth_eps_)

                plt.title('Iteration: %d' % (iteration_idx))
                # if use_js:
                #     if js_score.ndim>0:
                #         plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n JS(P||Q) = %1.3f'
                #                    % (iteration_idx, u, np.sqrt(variance), js_score[j]))
                #     else:
                #         plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n JS(P||Q) = %1.3f'
                #                    % (iteration_idx, u, np.sqrt(variance), js_score))
                # else:
                #     plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n mean err: %1.3f std err: %1.3f'
                #                   % (iteration_idx, u, np.sqrt(variance), depth_gt_lidar[j]-u, eps_min-np.sqrt(variance)))

                x_axis = np.arange(np.amin(x), np.amax(x), 0.01)

                if depth_only:
                    # expected depth
                    plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, expected_depth[j], 0.01))
                                , color=np.array([125, 45, 200])/255., linewidth=7)
                    # measured depth
                    plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], 0.01))
                                , color=np.array([0, 176, 80])/255., linewidth=7)
                else:
                    # LOS-based distribution
                    if use_js:
                        plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], eps_dynamic_[j]))
                                , color=np.array([239, 134, 0])/255., linewidth=7)
                        plt.fill_between(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], eps_dynamic_[j]))
                                , color=np.array([255, 238, 217])/255., linewidth=4)
                    else:
                        plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], depth_eps_))
                                , color=np.array([239, 134, 0])/255., linewidth=7)
                        plt.fill_between(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], depth_eps_))
                                , color=np.array([255, 238, 217])/255., linewidth=4)
                    
                    # sample distribution
                    plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, u, np.sqrt(variance)))
                                , color=np.array([125, 45, 200])/255., linewidth=5) # np.amax(y)
                    # goal distribution
                    plt.plot(x_axis, self.normalize_dist(norm.pdf(x_axis, depth_gt_lidar[j], eps_min)) 
                                , color=np.array([0, 176, 80])/255., linewidth=4)

                # sample
                plt.plot(x,y,'.', markersize=8, color=np.array([0, 112, 192])/255.) 
                # plt.plot(x,y_gt,'.', linewidth=0.5) 

                plt.xlabel("Dist. (m)")
                plt.ylabel("Predicted weight")
                plt.ylim([0, 1])
                # plt.legend(["Training distribution", "Sample distribution", "Goal distribution", "Sample weights"], loc ="upper center")
                
                os.makedirs(f"{self._settings.log_directory}/viz_loss", exist_ok=True)
                fname = f"{self._settings.log_directory}/viz_loss/iter_{self._global_step}.png"
                plt.savefig(fname)
                # plt.show()
    def normalize_dist(self, pdf):
        if np.amax(pdf) > 1:
            pdf = pdf / np.amax(pdf)
        return pdf


