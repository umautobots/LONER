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
import wandb
from torch.profiler import ProfilerActivity, profile
# for visualization
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

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

# TODO (hack): This is duplicated in KF Manager
class SampleAllocationStrategy(Enum):
    UNIFORM = 0
    ACTIVE = 1

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
    """ The Optimizer module is used to run iterations of the CLONeR Optimization.

    The KeyFrameManager supplies the Optimizer with a window of KeyFrame objects,
    which the Optimizer then uses to draw samples and iterate the optimization
    """

    ## Constructor
    # @param settings: Optimizer-specific settings. See the example settings for details.
    # @param calibration: Calibration-related settings. See the example settings for details
    # @param world_cube: The world cube pre-computed that is used to scale the world.
    # @param device: Which device to put the data on and run the optimizer on
    def __init__(self, settings: Settings, calibration: Settings, world_cube: WorldCube, device: int,
                 use_gt_poses: bool = False, sample_allocation_strategy: str = "UNIFORM"):
        self._settings = settings
        self._calibration = calibration
        self._device = device
        
        self._use_gt_poses = use_gt_poses

        self._sample_allocation_strategy = SampleAllocationStrategy[sample_allocation_strategy]

        opt_settings = settings.default_optimizer_settings
        self._optimization_settings = OptimizationSettings(
            opt_settings.stage, opt_settings.num_iterations, opt_settings.fix_poses,
            opt_settings.fix_sigma_mlp, opt_settings.fix_rgb_mlp)

        # We pre-create random numbers to lookup at runtime to save runtime.
        # This kills a lot of memory, but saves a ton of runtime
        # self._lidar_shuffled_indices = torch.randperm(MAX_POSSIBLE_LIDAR_RAYS)
        self._rgb_shuffled_indices = torch.randperm(
            calibration.camera_intrinsic.width * calibration.camera_intrinsic.height)

        self._model_config = settings.model_config

        self._scale_factor = world_cube.scale_factor

        self._data_prep_device = 'cpu' if settings.data_prep_on_cpu else self._device

        self._world_cube = world_cube.to(self._data_prep_device)


        self._ray_range = torch.Tensor(
            self._model_config.model.ray_range).to(self._data_prep_device)

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
        self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())

        self._cam_ray_directions = CameraRayDirections(calibration, device=self._data_prep_device)
        self._keyframe_count = 0
        self._global_step = 0


        self._keyframe_schedule = self._settings["keyframe_schedule"]

        # TODO: This breaks multiprocessing
        # self._wandb_mode = "online" if ENABLE_WANDB else "disabled"
        # self._wandb = wandb.init(project='cloner_slam', config=self._model_config,
        #                          save_code=True, mode=self._wandb_mode)

    ## Run one or more iterations of the optimizer, as specified by the stored settings
    # @param keyframe_window: The set of keyframes to use in the optimization.
    def iterate_optimizer(self, keyframe_window: List[KeyFrame]) -> float:
                
        # Look at the keyframe schedule and figure out which iteration schedule to use
        cumulative_kf_idx = 0
        for item in self._keyframe_schedule:
            kf_count = item["num_keyframes"]
            iteration_schedule = item["iteration_schedule"]

            cumulative_kf_idx += kf_count
            if cumulative_kf_idx >= self._keyframe_count + 1 or kf_count == -1:
                break

        num_its = sum(i["num_iterations"] for i in iteration_schedule)

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
            result = self._do_iterate_optimizer(keyframe_window, iteration_schedule, prof)
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
            result = self._do_iterate_optimizer(keyframe_window, iteration_schedule)
            end_time = time.time()
            elapsed_time = end_time - start_time

            log_file = f"{self._settings.log_directory}/timing.csv"
            with open(log_file, 'a+') as f:
                f.write(f"{num_its},{elapsed_time}\n")
            print(f"Elapsed Time: {end_time - start_time}. Per Iteration: {(end_time - start_time)/num_its}, Its/Sec: {1/((end_time - start_time)/num_its)}")
       
        self._keyframe_count += 1
        return result

    def _do_iterate_optimizer(self, keyframe_window: List[KeyFrame], iteration_schedule: dict, profiler: profile = None) -> float:

        if len(keyframe_window) == 1:
            keyframe_window[0].is_anchored = True

        # TODO (seth): this is temphack
        if len(iteration_schedule) > 1 and self._settings.skip_pose_refinement:
            iteration_schedule = iteration_schedule[1:]

        # For each iteration config, have a list of the losses
        losses_log = []

        for iteration_config in iteration_schedule:
            losses_log.append([])

            self._optimization_settings.freeze_poses = iteration_config["fix_poses"]
            
            if "latest_kf_only" in iteration_config:
                self._optimization_settings.latest_kf_only = iteration_config["latest_kf_only"]
            else:
                self._optimization_settings.latest_kf_only = False

            self._optimization_settings.freeze_rgb_mlp = iteration_config["fix_rgb_mlp"]
            self._optimization_settings.freeze_sigma_mlp = iteration_config["fix_sigma_mlp"]
            self._optimization_settings.num_iterations = iteration_config["num_iterations"]
            self._optimization_settings.stage = iteration_config["stage"]

            self._ray_sampler.update_occ_grid(self._occupancy_grid.detach())

            self._model.freeze_sigma_head(self._optimization_settings.freeze_sigma_mlp)
            self._model.freeze_rgb_head(self._optimization_settings.freeze_rgb_mlp)

            trainable_model_params = [
                p for p in self._model.parameters() if p.requires_grad]

            optimize_poses = not self._optimization_settings.freeze_poses

            if self._optimization_settings.latest_kf_only:
                most_recent_ts = -1
                for kf in keyframe_window:
                    if kf.get_start_time() > most_recent_ts:
                        most_recent_ts = kf.get_start_time()
                        most_recent_kf = kf

                active_keyframe_window = [most_recent_kf]
            else:
                active_keyframe_window = keyframe_window

            for kf in active_keyframe_window:
                if not kf.is_anchored:
                    kf.get_start_lidar_pose().set_fixed(not optimize_poses)
                    kf.get_end_lidar_pose().set_fixed(not optimize_poses)

            if optimize_poses:
            
                optimizable_poses = [kf.get_start_lidar_pose().get_pose_tensor() for kf in active_keyframe_window if not kf.is_anchored] \
                    + [kf.get_end_lidar_pose().get_pose_tensor()
                    for kf in active_keyframe_window if not kf.is_anchored]
                print(f"Num keyframes: {len(active_keyframe_window)}, Num Trainable Poses: {len(optimizable_poses)}")
                self._optimizer = torch.optim.Adam([{'params': trainable_model_params, 'lr': self._model_config.train.lrate_mlp},
                                            {'params': optimizable_poses, 'lr': self._model_config.train.lrate_pose}])

            else:
                self._optimizer = torch.optim.Adam(
                    trainable_model_params, lr=self._model_config.train.lrate_mlp)

            lrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._model_config.train.lrate_gamma)

            for it_idx in tqdm.tqdm(range(self._optimization_settings.num_iterations)):
                uniform_lidar_rays, uniform_lidar_depths = None, None
                uniform_camera_rays, uniform_camera_intensities = None, None
                
                # Bookkeeping for occ update
                # TODO: Find a better solution to this.
                self._results_lidar = None
        
                camera_samples, lidar_samples = None, None

                for kf_idx, kf in enumerate(active_keyframe_window):                
                    if self.should_enable_lidar():

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
                        
                        lidar_samples = (uniform_lidar_rays.to(self._device).float(), uniform_lidar_depths.to(self._device).float())
    
                    if self.should_enable_camera():
                        
                        # Get all the uniform samples first
                        start_idxs = torch.randint(len(self._rgb_shuffled_indices), (2,))

                        first_im_end_idx = start_idxs[0] + kf.num_uniform_rgb_samples
                        first_im_uniform_indices = self._rgb_shuffled_indices[start_idxs[0]:first_im_end_idx]
                        first_im_uniform_indices = first_im_uniform_indices % len(self._rgb_shuffled_indices)

                        second_im_end_idx = start_idxs[1] + kf.num_uniform_rgb_samples
                        second_im_uniform_indices = self._rgb_shuffled_indices[start_idxs[1]:second_im_end_idx]
                        second_im_uniform_indices = second_im_uniform_indices % len(self._rgb_shuffled_indices)

                        # Next based on strategy get the rest
                        if self._sample_allocation_strategy == SampleAllocationStrategy.UNIFORM:
                            start_idxs = torch.randint(len(self._rgb_shuffled_indices), (2,))

                            first_im_end_idx = start_idxs[0] + kf.num_strategy_rgb_samples

                            first_im_strategy_indices = self._rgb_shuffled_indices[start_idxs[0]:first_im_end_idx]

                            first_im_strategy_indices = first_im_strategy_indices % len(self._rgb_shuffled_indices)

                            second_im_end_idx = start_idxs[1] + kf.num_strategy_rgb_samples
                            second_im_strategy_indices = self._rgb_shuffled_indices[start_idxs[1]:second_im_end_idx]

                            second_im_strategy_indices = second_im_strategy_indices % len(self._rgb_shuffled_indices)

                        elif self._sample_allocation_strategy == SampleAllocationStrategy.ACTIVE:
                            first_im_strategy_indices = self._cam_ray_directions.sample_chunks(kf.loss_distribution, kf.num_strategy_rgb_samples.item()).flatten()
                            second_im_strategy_indices = self._cam_ray_directions.sample_chunks(kf.loss_distribution, kf.num_strategy_rgb_samples.item()).flatten()

                        first_im_indices = torch.cat((first_im_uniform_indices, first_im_strategy_indices))
                        second_im_indices = torch.cat((second_im_uniform_indices, second_im_strategy_indices))
                        
                        new_cam_rays, new_cam_intensities = kf.build_camera_rays(
                            first_im_indices, second_im_indices, self._ray_range,
                            self._cam_ray_directions, self._world_cube,
                            self._use_gt_poses,
                            self._settings.detach_rgb_from_poses)

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
                    
                        camera_samples = (uniform_camera_rays.to(self._device).float(), uniform_camera_intensities.to(self._device).float())

                loss = self.compute_loss(camera_samples, lidar_samples, self._optimization_settings.stage)

                losses_log[-1].append(loss.detach().cpu().item())

                if self._settings.debug.draw_comp_graph:
                    graph_dir = f"{self._settings.log_directory}/graphs"
                    os.makedirs(graph_dir, exist_ok=True)
                    loss_dot = torchviz.make_dot(loss)
                    loss_dot.format = "png"
                    loss_dot.render(directory=graph_dir, filename=f"iteration_{self._global_step}")

                loss.backward(retain_graph=False)
                self._optimizer.step()
                lrate_scheduler.step()

                self._optimizer.zero_grad(set_to_none=True)

                if self.should_enable_lidar() and \
                        self._global_step % self._model_config.model.occ_model.N_iters_acc == 0:
                    self._step_occupancy_grid()
                self._global_step += 1

                if profiler is not None:
                    profiler.step()


        if self._settings.debug.log_losses:
            graph_dir = f"{self._settings.log_directory}/losses/keyframe_{self._keyframe_count}"
            os.makedirs(graph_dir, exist_ok=True)
            for log_idx, log in enumerate(losses_log):
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
        return self._optimization_settings.stage in [2, 3] \
                and (not self._optimization_settings.freeze_rgb_mlp \
                     or (not self._settings.detach_rgb_from_poses \
                         and not self._optimization_settings.freeze_poses))

    ## For an 8x8 grid on the start image, compute the loss of each cell
    def compute_rgb_loss_distribution(self, keyframe: KeyFrame):
        # TODO: For simplicity, this only computes losses for the first image. Is this OK?
        chunk_indices = self._cam_ray_directions.sample_chunks()

        losses = []

        # TODO: Remove this for loop, do with pytorch operations instead
        for chunk in chunk_indices:
            cam_rays,cam_intensities = keyframe.build_camera_rays(
                            chunk, None, self._ray_range,
                            self._cam_ray_directions, self._world_cube,
                            self._use_gt_poses)

            cam_rays = cam_rays.to(self._device)
            cam_intensities = cam_intensities.to(self._device)

            chunk_loss = self.compute_loss((cam_rays,cam_intensities), None, True)
            losses.append(chunk_loss)

        losses = torch.stack(losses)
        return losses

    ## For the given camera and lidar rays, compute and return the differentiable loss
    def compute_loss(self, camera_samples: Tuple[torch.Tensor, torch.Tensor], 
                           lidar_samples: Tuple[torch.Tensor, torch.Tensor],
                           optimization_stage: int,
                           override_enables: bool = False) -> torch.Tensor:
        scale_factor = self._scale_factor.to(self._device).float()
        
        loss = 0
        wandb_logs = {}

        iteration_idx = self._global_step % self._optimization_settings.num_iterations
        # print('iteration_idx: ', iteration_idx)
        # print('self._optimization_settings.num_iterations: ', self._optimization_settings.num_iterations)

        if (override_enables or self.should_enable_lidar()) and lidar_samples is not None:

            if self._model_config.loss.decay_depth_lambda:
                depth_lambda = max(self._model_config.loss.depth_lambda * (self._model_config.train.decay_rate ** (
                    (self._global_step + 1) / (self._model_config.loss.depth_lambda_decay_steps))), self._model_config.loss.min_depth_lambda)
            else:
                depth_lambda = self._model_config.loss.depth_lambda
            
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

            self._lidar_depths_gt = lidar_depths * scale_factor # [N_rays, 1]
            weights_pred_lidar = self._results_lidar['weights_fine']
            
            # Compute JS divergence (also calculate when using fix depth_eps for visualization)
            mean = torch.sum(self._lidar_depth_samples_fine * weights_pred_lidar, axis=1) / (torch.sum(weights_pred_lidar, axis=1) + 1e-10) # weighted mean # [N_rays]
            var = torch.sum((self._lidar_depth_samples_fine-torch.unsqueeze(mean, dim=-1))**2 * weights_pred_lidar, axis=1) / (torch.sum(weights_pred_lidar, axis=1) + 1e-10) + 1e-10 # [N_rays]
            eps = torch.sqrt(var) # [N_rays]
            mean, var, eps = torch.unsqueeze(mean, 1), torch.unsqueeze(var, 1), torch.unsqueeze(eps, 1)
            eps_min = self._model_config.loss.min_depth_eps
            js_score = self.jsd_gauss(self._lidar_depths_gt, eps_min, mean, eps).squeeze()

            #print('self._model_config.loss.dynamic_depth_eps_JS: ', self._model_config.loss.dynamic_depth_eps_JS)
            if self._model_config.loss.dynamic_depth_eps_JS:
                # print('using JS divergence loss')
                min_js_score = self._model_config.loss.JS_loss.min_js_score
                max_js_score = self._model_config.loss.JS_loss.max_js_score
                alpha = self._model_config.loss.JS_loss.alpha
                js_score[js_score<min_js_score] = 0
                js_score[js_score>max_js_score] = max_js_score
                eps_dynamic = eps_min*(1+(alpha * js_score))
                eps_dynamic = torch.unsqueeze(eps_dynamic, dim=-1).detach()
                weights_gt_lidar = get_weights_gt(self._lidar_depth_samples_fine, self._lidar_depths_gt, eps=eps_dynamic) # [N_rays, N_samples]

                if self._model_config.loss.viz_loss:
                    # viz_idx = np.where(js_score.detach().cpu().numpy() == max_js_score)[0] # show rays that haven't converged
                    viz_idx = np.array([0]) # show the first ray
                    self.viz_loss(iteration_idx, viz_idx, opaque_rays.detach().cpu().numpy(), weights_gt_lidar.detach().cpu().numpy(), weights_pred_lidar.detach().cpu().numpy(), \
                                mean.detach().cpu().numpy(), var.detach().cpu().numpy(), js_score.detach().cpu().numpy(), \
                                self._lidar_depth_samples_fine.detach().cpu().numpy(), self._lidar_depths_gt.detach().cpu().numpy(), eps_dynamic.detach().cpu().numpy())
            else:
                if self._model_config.loss.decay_depth_eps:
                    depth_eps = max(self._model_config.loss.depth_eps * (self._model_config.train.decay_rate ** (
                                    iteration_idx / (self._model_config.loss.depth_eps_decay_steps))), self._model_config.loss.min_depth_eps)
                else:
                    depth_eps = self._model_config.loss.depth_eps
                weights_gt_lidar = get_weights_gt(
                    self._lidar_depth_samples_fine, self._lidar_depths_gt, eps=depth_eps)
                
                if self._model_config.loss.viz_loss:
                    # viz_idx = np.where(js_score.detach().cpu().numpy() == max_js_score)[0] # show rays that haven't converged
                    viz_idx = np.array([0]) # show the first ray
                    self.viz_loss(iteration_idx, viz_idx, opaque_rays.detach().cpu().numpy(), weights_gt_lidar.detach().cpu().numpy(), weights_pred_lidar.detach().cpu().numpy(), \
                                    mean.detach().cpu().numpy(), var.detach().cpu().numpy(), js_score.detach().cpu().numpy(), \
                                    self._lidar_depth_samples_fine.detach().cpu().numpy(), self._lidar_depths_gt.detach().cpu().numpy(), depth_eps)
            
            weights_gt_lidar[~opaque_rays, :] = 0

            # # [NEW]
            # if optimization_stage != 1:
            #     # print('use step loss')
            #     # print('lidar_depths:', lidar_depths)
            #     mask = self._lidar_depth_samples_fine > (self._lidar_depths_gt + 0.) # N (m) tail
            #     weights_pred_lidar = weights_pred_lidar * torch.logical_not(mask)
            #     weights_gt_lidar = weights_gt_lidar * torch.logical_not(mask) * 10.

            #     # j = 0
            #     # x = self._lidar_depth_samples_fine[j].detach().cpu().numpy()
            #     # y = weights_pred_lidar[j].detach().cpu().numpy()
            #     # y_gt = weights_gt_lidar[j].detach().cpu().numpy()
            #     # mask_ = mask[j].detach().cpu().numpy()
            #     # plt.figure(figsize=(15, 10))
            #     # plt.plot(x, y, '.', linewidth=0.5) 
            #     # plt.plot(x, y_gt, '.', linewidth=0.5) 
            #     # plt.plot(x, mask_, '.', linewidth=0.5) 
            #     # plt.show()
            #     # print('torch.count_nonzero(mask): ', torch.count_nonzero(mask))
            #     depth_loss_los_fine = nn.functional.l1_loss(
            #         weights_pred_lidar, weights_gt_lidar, reduction='sum') / (torch.numel(mask)-torch.count_nonzero(mask))
            # else:
            #     depth_loss_los_fine = nn.functional.l1_loss(
            #         weights_pred_lidar, weights_gt_lidar)
            
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

        if (override_enables or self.should_enable_camera()) and camera_samples is not None:
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


    def kld_gauss(self, u1, s1, u2, s2):
        # general KL two Gaussians
        # u2, s2 often N(0,1)
        # https://stats.stackexchange.com/questions/7440/ +
        # kl-divergence-between-two-univariate-gaussians
        # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
        v1 = s1 * s1
        v2 = s2 * s2
        a = torch.log(s2/s1) 
        num = v1 + (u1 - u2)**2
        den = 2 * v2
        b = num / den
        return a + b - 0.5
    
    def jsd_gauss(self, u1, s1, u2, s2):
        um = 0.5 * (u1+u2)
        sm = 0.5 * torch.sqrt(s1**2+s2**2)
        return 0.5 * self.kld_gauss(u1, s1, um, sm) + 0.5 * self.kld_gauss(u2, s2, um, sm)

    def viz_loss(self, i, viz_idx: np.ndarray, opaque_rays: np.ndarray, weights_gt_lidar: np.ndarray, weights_pred_lidar: np.ndarray,
                 mean: np.ndarray, var: np.ndarray, js_score: np.ndarray, s_vals_lidar: np.ndarray,
            depth_gt_lidar: np.ndarray, eps_: np.ndarray)->None:
        if i > 0:
            # max_js_ids = np.where(js_score == self._model_config.loss.JS_loss.max_js_score)[0]
            # opaque_ids = np.where(opaque_rays == True)[0]
            # print('maxjs_count: ', len(np.intersect1d(max_js_ids, opaque_ids)) )
            # return
            for j in viz_idx:
                if not opaque_rays[j]:
                    continue
                print('ray idx:', j)
                x = s_vals_lidar[j]
                y = weights_pred_lidar[j]
                y_gt = weights_gt_lidar[j]
                u = mean[j]
                variance = var[j]
                print("u: ", u, " var: ", variance)
                plt.figure(figsize=(15, 10))
                plt.plot(x,y,'.', linewidth=0.5) 
                plt.plot(x,y_gt,'.', linewidth=0.5) 

                depth_gt_lidar = np.array(np.squeeze(depth_gt_lidar)).reshape((-1))
                eps_min = self._model_config.loss.min_depth_eps
                if self._model_config.loss.dynamic_depth_eps_JS:
                    eps_dynamic_ = eps_
                    print("u_gt:", depth_gt_lidar[j], "eps: ", eps_dynamic_[j])
                else:
                    depth_eps_ = eps_
                    print("u_gt:", depth_gt_lidar[j], "eps: ", depth_eps_)
                if self._model_config.loss.dynamic_depth_eps_JS:
                    if js_score.ndim>0:
                        plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n JS(P||Q) = %1.3f' % (i, u, np.sqrt(variance), js_score[j]))
                    else:
                        plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n JS(P||Q) = %1.3f' % (i, u, np.sqrt(variance), js_score))
                else:
                    plt.title('Iter: %d\n mean: %1.3f std: %1.3f\n mean err: %1.3f std err: %1.3f' % (i, u, np.sqrt(variance), depth_gt_lidar[j]-u, eps_min-np.sqrt(variance)))

                x_axis = np.arange(np.amin(x), np.amax(x), 0.01)
                plt.plot(x_axis, norm.pdf(x_axis, u, np.sqrt(variance)) * (0.5/np.amax(norm.pdf(x_axis, u, np.sqrt(variance)))), '-m', linewidth=2) # np.amax(y)
                #plt.plot(x_axis, norm.pdf(x_axis, depth_gt_lidar[0], depth_eps) * np.amax(y), '-g', linewidth=3)
                plt.plot(x_axis, norm.pdf(x_axis, depth_gt_lidar[j], eps_min) * (0.5/np.amax(norm.pdf(x_axis, depth_gt_lidar[j], eps_min))), '-g', linewidth=2)
                if self._model_config.loss.dynamic_depth_eps_JS:
                    plt.plot(x_axis, norm.pdf(x_axis, depth_gt_lidar[j], eps_dynamic_[j]) * (0.5/np.amax(norm.pdf(x_axis, depth_gt_lidar[j], eps_dynamic_[j]))), '-r', linewidth=2)
                else:
                    plt.plot(x_axis, norm.pdf(x_axis, depth_gt_lidar[j], depth_eps_) * (0.5/np.amax(norm.pdf(x_axis, depth_gt_lidar[j], depth_eps_))), '-r', linewidth=2)
                plt.xlabel("Dist. (m)")
                plt.ylabel("Predicted weight")
                plt.legend(["Sample results", "Sample gt", "Sample distribution", "Goal distribution", "Training distribution"], loc ="upper center")
                plt.show()
