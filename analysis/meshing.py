#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pathlib
import pickle
import re
import sys
import torch.multiprocessing as mp
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

import open3d as o3d

from render_utils import *

from src.common.pose import Pose
from src.common.pose_utils import WorldCube
from src.common.ray_utils import CameraRayDirections, LidarRayDirections
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler, UniformRaySampler

CHUNK_SIZE = 512

from src.common.sensors import LidarScan
from analysis.mesher import Mesher

assert torch.cuda.is_available(), 'Unable to find GPU'
import yaml
from pathlib import Path


parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, nargs="+", help="folder in outputs with all results")
parser.add_argument("configuration_path")
# parser.add_argument("sequence", type=str, default="canteen", help="sequence name. Used to decide meshing bound [canteen | mcr]")
parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--resolution", type=float, default=0.1, help="grid resolution (m)")
parser.add_argument("--threshold", type=float, default=0.0, help="threshold for sigma MLP output. default as 0")

parser.add_argument("--color", default=False, dest="color", action="store_true")
parser.add_argument("--viz", default=False, dest="viz", action="store_true")
parser.add_argument("--save", default=False, dest="save", action="store_true")
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")

parser.add_argument("--use_lidar_fov_mask", default=False, dest="use_lidar_fov_mask", action="store_true")
parser.add_argument("--use_convex_hull_mask", default=False, dest="use_convex_hull_mask", action="store_true")
parser.add_argument("--use_lidar_pointcloud_mask", default=False, dest="use_lidar_pointcloud_mask", action="store_true")
parser.add_argument("--use_occ_mask", default=False, dest="use_occ_mask", action="store_true")
parser.add_argument("--retrain_occ", default=False, action="store_true")
parser.add_argument("--max_range", type=float, default=None)
parser.add_argument("--use_weights", default=False, action="store_true")
parser.add_argument("--level", type=float, default=0)
parser.add_argument("--skip_step", type=int, default=15)
parser.add_argument("--var_threshold", type=float, default=None)


parser.add_argument("--color_render_from_ray", default=False, dest="color_render_from_ray", action="store_true")


args = parser.parse_args()

def build_mesh(exp_dir):
    checkpoints = os.listdir(f"{exp_dir}/checkpoints")

    if not (args.viz or args.save):
        raise RuntimeError("Either visualize or save.")

    with open(args.configuration_path) as config_file:
        config = yaml.full_load(config_file)
    rosbag_path = Path(os.path.expanduser(config["dataset"]))

    x_min, x_max = config['meshing_bounding_box']['x']
    y_min, y_max = config['meshing_bounding_box']['y']
    z_min, z_max = config['meshing_bounding_box']['z']
    meshing_bound = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    resolution = args.resolution
    sigma_only = not args.color

    use_lidar_fov_mask = args.use_lidar_fov_mask
    use_convex_hull_mask = args.use_convex_hull_mask
    use_lidar_pointcloud_mask = args.use_lidar_pointcloud_mask
    use_occ_mask = args.use_occ_mask
    threshold = args.threshold

    if args.color_render_from_ray:
        color_mesh_extraction_method = 'render_ray'
    else:
        color_mesh_extraction_method = 'direct_point_query'

    if args.ckpt_id is None:
        #https://stackoverflow.com/a/2669120
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
        args.ckpt_id = checkpoint.split('.')[0]
    elif args.ckpt_id=='final':
        checkpoint = f"final.tar"
    else:
        checkpoint = f"ckpt_{args.ckpt_id}.tar"
    checkpoint_path = pathlib.Path(f"{exp_dir}/checkpoints/{checkpoint}")

    os.makedirs(f"{exp_dir}/meshing/resolution_{resolution}/", exist_ok=True)
    mesh_out_file=f"{exp_dir}/meshing/resolution_{resolution}/ckpt_{args.ckpt_id}.ply"

    # override any params loaded from yaml
    with open(f"{exp_dir}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)
    if args.debug:
        full_config['debug'] = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    _DEVICE = torch.device(full_config.mapper.device)
    print('_DEVICE', _DEVICE)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not checkpoint_path.exists():
        print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
        exit()

    occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
    assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
    scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
    shift = full_config.world_cube.shift
    world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

    cfg = full_config.mapper.optimizer.model_config
    ray_range = cfg.data.ray_range
    if args.max_range is not None:
        ray_range = (ray_range[0], args.max_range)

    print(f'Loading checkpoint from: {checkpoint_path}') 
    ckpt = torch.load(str(checkpoint_path))

    model_config = full_config.mapper.optimizer.model_config.model
    model = Model(model_config).to(_DEVICE)
    model.load_state_dict(ckpt['network_state_dict']) 


    def write_pcd(occ_grid, fname):
        occ_sigma_np = occ_grid.squeeze().cpu().detach().numpy()
        if occ_sigma_np.sum() > 1:
            occ_probs = 1. / (1 + np.exp(-occ_sigma_np))
            occ_probs = (510 *  (occ_probs.clip(0.5, 1.0) - 0.5)).astype(np.uint8).reshape(-1)
            nonzero_indices = occ_probs.nonzero()
            x_ = np.arange(cfg.model.occ_model.voxel_size)
            x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
            X = np.stack([x.reshape(-1)[nonzero_indices], y.reshape(-1)[nonzero_indices], -z.reshape(-1)[nonzero_indices], occ_probs[nonzero_indices]], axis=1)

        with open(fname, 'w') as f:
            if X.shape[0] <= 4:
                X = X.T
                assert X.shape[0] > 4, f"Too few points or wrong shape of pcd file."
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 1\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {X.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {X.shape[0]}\n")
            f.write("DATA ascii\n")
            for pt in X:
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {pt[3]}\n")

    occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)

    occupancy_grid = occ_model()
    ray_sampler = OccGridRaySampler()
    ray_sampler.update_occ_grid(occupancy_grid.detach())
    occ_model.load_state_dict(ckpt['occ_model_state_dict'])

    # rosbag_path = full_config.dataset_path
    lidar_topic = full_config.system.ros_names.lidar
    ray_range = full_config.mapper.optimizer.model_config.data.ray_range
    mesher = Mesher(model, ckpt, world_cube, ray_range, rosbag_path=rosbag_path, lidar_topic=lidar_topic,  level_set=args.level, 
                    resolution=resolution, marching_cubes_bound=meshing_bound, points_batch_size=500000,
                    lidar_vertical_fov=config["lidar_vertical_fov"])
    mesh_o3d, mesh_lidar_frames = mesher.get_mesh(_DEVICE, ray_sampler, occupancy_grid, occ_voxel_size=occ_model_config.voxel_size, sigma_only=sigma_only, threshold=threshold, 
                                                    use_lidar_fov_mask=use_lidar_fov_mask, use_convex_hull_mask=use_convex_hull_mask,use_lidar_pointcloud_mask=use_lidar_pointcloud_mask, use_occ_mask=use_occ_mask,
                                                    color_mesh_extraction_method=color_mesh_extraction_method,
                                                    use_weights = args.use_weights,
                                                    skip_step = args.skip_step,
                                                    var_threshold=args.var_threshold)
    mesh_o3d.compute_vertex_normals()

    if args.viz:
        # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=np.squeeze(shift))
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        o3d.visualization.draw_geometries([mesh_o3d, origin_frame],
                                        mesh_show_back_face=True, mesh_show_wireframe=False)

    if args.save:
        print('store mesh at: ', mesh_out_file)
        o3d.io.write_triangle_mesh(mesh_out_file, mesh_o3d, compressed=False, write_vertex_colors=True, 
                                write_triangle_uvs=False, print_progress=True)

def _gpu_worker(job_queue: mp.Queue):
    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            break
        exp_dir = data
        build_mesh(exp_dir)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    if len(args.experiment_directory) == 1:
        build_mesh(args.experiment_directory[0])
        exit(0)

    job_queue = mp.Queue()
    for exp_dir in args.experiment_directory:
        job_queue.put(exp_dir)

    for _ in range(torch.cuda.device_count()):
        job_queue.put(None)
        
    gpu_worker_processes = []
    for gpu_id in range(torch.cuda.device_count()):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(job_queue,)))
        gpu_worker_processes[-1].start()

    # Sync
    for process in gpu_worker_processes:
        process.join()