import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.profiler
import torchviz
import tqdm
import wandb
from torch.profiler import ProfilerActivity, profile

from common.pose_utils import WorldCube
from common.ray_utils import CameraRayDirections, rays_to_pcd
from common.settings import Settings
from mapping.keyframe import KeyFrame
from models.losses import (get_logits_grad, get_weights_gt, img_to_mse,
                           mse_to_psnr)
from models.model_tcnn import Model, OccupancyGridModel
from models.ray_sampling import OccGridRaySampler

# Used for pre-creating random indices
MAX_POSSIBLE_LIDAR_RAYS = int(1e7)

ENABLE_WANDB = False

class SampleStrategy(Enum):
    UNIFORM = 0

# Note: This is a bit of a deviation from how the rest of the code handles settings.
# But it makes sense here to be a bit extra explicit since we'll change these often


@dataclass
class OptimizationSettings:
    """ OptimizationSettings is a simple container for parameters for the optimizer
    """

    stage: int = 3
    num_iterations: int = 1
    freeze_poses: bool = False  # Fix the map, only optimize poses
    freeze_sigma_mlp: bool = False
    freeze_rgb_mlp: bool = False


class Optimizer:
    """ The Optimizer module is used to run iterations of the CLONeR Optimization.

    The KeyFrameManager supplies the Optimizer with a window of KeyFrame objects,
    which the Optimizer then uses to draw samples and iterate the optimization
    """

    # Constructor
    # @param settings: Optimizer-specific settings. See the example settings for details.
    # @param calibration: Calibration-related settings. See the example settings for details
    # @param world_cube: The world cube pre-computed that is used to scale the world.
    # @param device: Which device to put the data on and run the optimizer on
    def __init__(self, settings: Settings, calibration: Settings, world_cube: WorldCube, device: int,
                 use_gt_poses: bool = False):
        self._settings = settings
        self._calibration = calibration
        self._device = device
        
        self._use_gt_poses = use_gt_poses

        opt_settings = settings.default_optimizer_settings
        self._optimization_settings = OptimizationSettings(
            opt_settings.stage, opt_settings.num_iterations, opt_settings.fix_poses,
            opt_settings.fix_sigma_mlp, opt_settings.fix_rgb_mlp)

        self._sample_strategy = SampleStrategy[settings.sample_strategy]

        if self._sample_strategy != SampleStrategy.UNIFORM:
            raise ValueError(
                "Invalid sample strategy: Only UNIFORM is currently supported")

        # We pre-create random numbers to lookup at runtime to save runtime.
        # This kills a lot of memory, but saves a ton of runtime
        # self._lidar_shuffled_indices = torch.randperm(MAX_POSSIBLE_LIDAR_RAYS)
        self._rgb_shuffled_indices = torch.randperm(
            calibration.camera_intrinsic.width * calibration.camera_intrinsic.height)

        self._model_config = settings.model_config

        self._scale_factor = world_cube.scale_factor
        self._world_cube = world_cube

        self._ray_range = torch.Tensor(
            self._model_config.model.ray_range) #.to(self._device)

        # Main Model
        self._model = Model(self._model_config.model)

        # Occupancy grid
        self._occupancy_grid_model = OccupancyGridModel(
            self._model_config.model.occ_model).to(self._device)
        self._occupancy_grid = self._occupancy_grid_model()

        occ_grid_parameters = [
            p for p in self._occupancy_grid_model.parameters() if p.requires_grad]
        self._occupancy_grid_optimizer = torch.optim.SGD(
            occ_grid_parameters, lr=self._model_config.model.occ_model.lr)

        self._ray_sampler = OccGridRaySampler()


        self._cam_ray_directions = CameraRayDirections(calibration, device='cpu')
        self._keyframe_count = 0
        self._global_step = 0

        self._iteration_schedule = [(i["num_keyframes"], i["iteration_count"]) for i in self._settings["iteration_schedule"]]

        # TODO: This breaks multiprocessing
        # self._wandb_mode = "online" if ENABLE_WANDB else "disabled"
        # self._wandb = wandb.init(project='cloner_slam', config=self._model_config,
        #                          save_code=True, mode=self._wandb_mode)

    # Run one or more iterations of the optimizer, as specified by the stored settings
    # @param keyframe_window: The set of keyframes to use in the optimization.
    def iterate_optimizer(self, keyframe_window: List[KeyFrame]) -> float:
                
        cumulative_kf_idx = 0
        for kf_count, num_its in self._iteration_schedule:
            cumulative_kf_idx += kf_count
            if cumulative_kf_idx >= self._keyframe_count + 1 or kf_count == -1:
                break

        self._optimization_settings.num_iterations = num_its

        if self._settings.debug.profile:
            prof_dir = f"{self._settings.log_directory}/profile"
            os.makedirs(prof_dir, exist_ok=True)

            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         profile_memory=True,
                         record_shapes=True,
                         with_stack=True,
                         with_modules=True,
                         schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_its - 2),
                         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{prof_dir}/tensorboard/"))
            
            prof.start()
            start_time = time.time()
            result = self._do_iterate_optimizer(keyframe_window, prof)
            end_time = time.time()
            prof.stop()

            print(f"Elapsed Time: {end_time - start_time}. Per Iteration: {(end_time - start_time)/num_its}, Its/Sec: {1/((end_time - start_time)/num_its)}")

            # with open(f"{prof_dir}/stats.txt", 'a+') as prof_file:
            #     prof_file.write(f"{prof.key_averages().table(sort_by='self_cuda_time_total')}\n")

            # prof.export_stacks(f"{prof_dir}/step_{self._global_step}_cuda.stacks", metric="self_cuda_time_total")
            # prof.export_stacks(f"{prof_dir}/step_{self._global_step}_cpu.stacks", metric="self_cpu_time_total")
            # prof.export_chrome_trace(f"{prof_dir}/step_{self._global_step}_trace.json")            
        else:          
            start_time = time.time()
            result = self._do_iterate_optimizer(keyframe_window)
            end_time = time.time()
            print(f"Elapsed Time: {end_time - start_time}. Per Iteration: {(end_time - start_time)/num_its}, Its/Sec: {1/((end_time - start_time)/num_its)}")

        self._keyframe_count += 1
        return result

    def _do_iterate_optimizer(self, keyframe_window: List[KeyFrame], profiler: profile = None) -> float:
        self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())

        self._model.freeze_sigma_head(not self.should_enable_lidar())
        self._model.freeze_rgb_head(not self.should_enable_camera())

        trainable_model_params = [
            p for p in self._model.parameters() if p.requires_grad]

        optimize_poses = not self._optimization_settings.freeze_poses

        for kf in keyframe_window:
            kf.get_start_lidar_pose().set_fixed(not optimize_poses)
            kf.get_end_lidar_pose().set_fixed(not optimize_poses)

        if optimize_poses:
            trainable_poses = [kf.get_start_lidar_pose().get_pose_tensor() for kf in keyframe_window] \
                + [kf.get_end_lidar_pose().get_pose_tensor()
                   for kf in keyframe_window]
            print(f"Num keyframes: {len(keyframe_window)}, Num Trainable Poses: {len(trainable_poses)}")
            self._optimizer = torch.optim.Adam([{'params': trainable_model_params, 'lr': self._model_config.train.lrate_mlp},
                                          {'params': trainable_poses, 'lr': self._model_config.train.lrate_pose}])
        else:
            self._optimizer = torch.optim.Adam(
                trainable_model_params, lr=self._model_config.train.lrate_mlp)
        
        for it_idx in tqdm.tqdm(range(self._optimization_settings.num_iterations)):
            uniform_lidar_rays, uniform_lidar_depths = None, None
            uniform_camera_rays, uniform_camera_intensities = None, None
            
            # Bookkeeping for occ update
            # TODO: Find a better solution to this.
            self._results_lidar = None

            for kf_idx, kf in enumerate(keyframe_window):                
                if self.should_enable_lidar():
                    # lidar_start_idx = torch.randint(
                    #     MAX_POSSIBLE_LIDAR_RAYS, (1,))
                    # # TODO: This addition will NOT work for non-uniform sampling
                    # lidar_end_idx = lidar_start_idx + kf.num_uniform_lidar_samples
                    # lidar_indices = self._lidar_shuffled_indices[lidar_start_idx:lidar_end_idx]

                    # if kf.num_uniform_lidar_samples > len(kf.get_lidar_scan()):
                    #     print(
                    #         "Warning: Dropping lidar points since too many were requested")
                    #     lidar_end_idx = lidar_start_idx + \
                    #         len(kf.get_lidar_scan())

                    # # lidar_indices was roughly estimated using some way-too-big indices
                    # lidar_indices = lidar_indices % len(kf.get_lidar_scan())

                    lidar_indices = torch.randint(len(kf.get_lidar_scan()), (kf.num_uniform_lidar_samples,))

                    new_rays, new_depths = kf.build_lidar_rays(lidar_indices, self._ray_range, self._world_cube, self._use_gt_poses)
                
                    if self._settings.debug.write_ray_point_clouds:
                        os.makedirs(f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_rays", exist_ok=True)
                        os.makedirs(f"{self._settings['log_directory']}/rays//lidar/kf_{kf_idx}_origins", exist_ok=True)
                        rays_fname = f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_rays/rays_{self._global_step}_{it_idx}.pcd"
                        origins_fname = f"{self._settings['log_directory']}/rays/lidar/kf_{kf_idx}_origins/origins_{self._global_step}_{it_idx}.pcd"
                        rays_to_pcd(new_rays, new_depths, rays_fname, origins_fname)

                    if uniform_lidar_rays is None:
                        uniform_lidar_rays = new_rays
                        uniform_lidar_depths = new_depths
                    else:
                        uniform_lidar_rays = torch.vstack((uniform_lidar_rays, new_rays))
                        uniform_lidar_depths = torch.vstack((uniform_lidar_depths, new_depths))
                        
                if self.should_enable_camera():
                    start_idxs = torch.randint(
                        len(self._rgb_shuffled_indices), (2,))

                    # TODO: This addition will NOT work for non-uniform sampling
                    first_im_end_idx = start_idxs[0] + \
                        kf.num_uniform_rgb_samples

                    # autopep8: off
                    first_im_indices = self._rgb_shuffled_indices[start_idxs[0]:first_im_end_idx]

                    first_im_indices = first_im_indices % len(
                        self._rgb_shuffled_indices)

                    # TODO: This addition will NOT work for non-uniform sampling
                    second_im_end_idx = start_idxs[1] + \
                        kf.num_uniform_rgb_samples
                    second_im_indices = self._rgb_shuffled_indices[start_idxs[1]:second_im_end_idx]
                    # autopep8: on

                    second_im_indices = second_im_indices % len(
                        self._rgb_shuffled_indices)

                    new_cam_rays, new_cam_intensities = kf.build_camera_rays(
                        first_im_indices, second_im_indices, self._ray_range,
                        self._cam_ray_directions, self._world_cube,
                        self._use_gt_poses)

                    if uniform_camera_rays is None:
                        uniform_camera_rays, uniform_camera_intensities = new_cam_rays, new_cam_intensities
                    else:
                        uniform_camera_rays = torch.vstack((uniform_camera_rays, new_cam_rays))
                        uniform_camera_intensities = torch.vstack((uniform_camera_intensities, new_cam_intensities))

                    if self._settings.debug.write_ray_point_clouds:
                        os.makedirs(f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_rays", exist_ok=True)
                        os.makedirs(f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_origins", exist_ok=True)
                        rays_fname = f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_rays/rays_{self._global_step}_{it_idx}.pcd"
                        origins_fname = f"{self._settings['log_directory']}/rays/camera/kf_{kf_idx}_origins/origins_{self._global_step}_{it_idx}.pcd"
                        rays_to_pcd(new_cam_rays, torch.ones_like(new_cam_rays[:,0]) * .1, rays_fname, origins_fname, new_cam_intensities)
                
            lidar_samples = (uniform_lidar_rays.to(self._device).float(), uniform_lidar_depths.to(self._device).float())
            camera_samples = (uniform_camera_rays.to(self._device).float(), uniform_camera_intensities.to(self._device).float())

            loss = self.compute_loss(camera_samples, lidar_samples)

            if self._settings.debug.draw_comp_graph:
                loss_dot = torchviz.make_dot(loss)
                loss_dot.format = "png"
                loss_dot.render(directory="../graphs", filename=f"iteration_{self._global_step}")

            loss.backward(retain_graph=False)
            self._optimizer.step()

            self._optimizer.zero_grad(set_to_none=True)

            # TODO: Active Sampling


            if self.should_enable_lidar() and \
                    self._global_step % self._model_config.model.occ_model.N_iters_acc == 0:
                self._step_occupancy_grid()
            self._global_step += 1

            if profiler is not None:
                profiler.step()
    # Returns whether or not the lidar should be used, as indicated by the settings
    def should_enable_lidar(self) -> bool:
        return self._optimization_settings.stage in [1, 3] \
                    and not self._optimization_settings.freeze_sigma_mlp

    # Returns whether or not the camera should be use, as indicated by the settings

    def should_enable_camera(self) -> bool:
        return self._optimization_settings.stage in [2, 3] \
                and not self._optimization_settings.freeze_rgb_mlp

    # For the given camera and lidar rays, compute and return the differentiable loss
    def compute_loss(self, camera_samples: Tuple[torch.Tensor, torch.Tensor], 
                           lidar_samples: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        scale_factor = self._scale_factor.to(self._device).float()
        
        loss = 0
        wandb_logs = {}

        iteration_idx = self._global_step % self._optimization_settings.num_iterations
        # TODO: Update
        if self._model_config.loss.decay_depth_eps:
            depth_eps = max(self._model_config.loss.depth_eps * (self._model_config.train.decay_rate ** (
                            iteration_idx / (self._model_config.loss.depth_eps_decay_steps))), self._model_config.loss.min_depth_eps)
        else:
            depth_eps = self._model_config.loss.depth_eps

        if self._model_config.loss.decay_depth_lambda:
            depth_lambda = max(self._model_config.loss.depth_lambda * (self._model_config.train.decay_rate ** (
                (self._global_step + 1) / (self._model_config.loss.depth_lambda_decay_steps))), self._model_config.loss.min_depth_lambda)
        else:
            depth_lambda = self._model_config.loss.depth_lambda

        if self.should_enable_lidar():
            assert lidar_samples is not None, "Got None lidar_samples with lidar enabled"

            # rays = [origin, direction, viewdir, <ignore>, near limit, far limit]
            lidar_rays, lidar_depths = lidar_samples

            lidar_rays = lidar_rays.reshape(-1, lidar_rays.shape[-1])
            lidar_depths = lidar_depths.reshape(-1, 1)

            opaque_rays = (lidar_depths > 0)[..., 0]

            # Rendering lidar rays. Results need to be in class for occ update to happen
            self._results_lidar = self._model(
                lidar_rays, self._ray_sampler, scale_factor, camera=False)

            # (N_rays, N_samples)
            # Depths along ray
            self._lidar_depth_samples_fine = self._results_lidar['samples_fine'] * \
                scale_factor

            self._lidar_depths_gt = lidar_depths * scale_factor
            weights_pred_lidar = self._results_lidar['weights_fine']

            weights_gt_lidar = get_weights_gt(
                self._lidar_depth_samples_fine, self._lidar_depths_gt, eps=depth_eps)
            weights_gt_lidar[~opaque_rays, :] = 0

            depth_loss_los_fine = nn.functional.l1_loss(
                weights_pred_lidar, weights_gt_lidar)

            loss += depth_lambda * depth_loss_los_fine
            wandb_logs['loss_lidar_los'] = depth_loss_los_fine.item()

            depth_euc_fine = self._results_lidar['depth_fine'].unsqueeze(
                1) * scale_factor
            depth_loss_fine = nn.functional.mse_loss(
                depth_euc_fine[opaque_rays, 0], self._lidar_depths_gt[opaque_rays, 0])

            loss += self._model_config.loss.term_lambda * depth_loss_fine
            wandb_logs['loss_lidar_term'] = depth_loss_fine.item()

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

            if self._model_config.loss.decay_depth_lambda:
                depth_lambda = max(self._model_config.loss.depth_lambda * (self._model_config.train.decay_rate ** (
                    (self._global_step + 1) / (self._model_config.loss.depth_lambda_decay_steps))), self._model_config.loss.min_depth_lambda)
            if self._model_config.loss.decay_depth_eps:
                depth_eps = max(self._model_config.loss.depth_eps * (self._model_config.train.decay_rate ** (
                    (self._global_step + 1) / (self._model_config.loss.depth_eps_decay_steps))), self._model_config.loss.min_depth_eps)

            wandb_logs['depth_lambda'] = depth_lambda
            wandb_logs['depth_eps'] = depth_eps

        if self.should_enable_camera():
            assert camera_samples is not None, "Got None camera_samples with camera enabled"

            # camera_samples is organized as [cam_rays, cam_intensities]
            # cam_rays = [origin, direction, viewdir, ray_i_grid, ray_j_grid, near limit, far limit]
            cam_rays, cam_intensities = camera_samples

            cam_intensities = cam_intensities.detach()

            cam_rays = cam_rays.reshape(-1, cam_rays.shape[-1])
            cam_intensities = cam_intensities.reshape(
                -1, self._model_config.model.num_colors)

            results_cam = self._model(
                cam_rays, self._ray_sampler, scale_factor)

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

        # if self._global_step % self._model_config.log.i_log == 0:
        #     wandb.log(wandb_logs, commit=False)

        # wandb.log({}, commit=True)
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
