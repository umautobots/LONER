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
from src.common.pose_utils import WorldCube
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler
from analysis.mesher import Mesher
assert torch.cuda.is_available(), 'Unable to find GPU'
import yaml
from pathlib import Path

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, nargs="+", help="folder in outputs with all results")
parser.add_argument("configuration_path")
parser.add_argument("--debug", default=False, dest="debug", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--resolution", type=float, default=0.1, help="grid resolution (m)")
parser.add_argument("--max_range", type=float, default=None)
parser.add_argument("--level", type=float, default=0)
parser.add_argument("--skip_step", type=int, default=15)
parser.add_argument("--var_threshold", type=float, default=None)

parser.add_argument("--viz", default=False, dest="viz", action="store_true")
parser.add_argument("--save", default=False, dest="save", action="store_true")

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

    occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)

    occupancy_grid = occ_model()
    ray_sampler = OccGridRaySampler()
    ray_sampler.update_occ_grid(occupancy_grid.detach())
    occ_model.load_state_dict(ckpt['occ_model_state_dict'])

    ray_range = full_config.mapper.optimizer.model_config.data.ray_range
    mesher = Mesher(model, ckpt, world_cube, ray_range,  level_set=args.level, 
                    resolution=resolution, marching_cubes_bound=meshing_bound, points_batch_size=500000,
                    lidar_vertical_fov=config["lidar_vertical_fov"])
    mesh_o3d = mesher.get_mesh(_DEVICE, ray_sampler, skip_step=args.skip_step, var_threshold=args.var_threshold)
    mesh_o3d.compute_vertex_normals()

    if args.viz:
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        o3d.visualization.draw_geometries([mesh_o3d, origin_frame],
                                        mesh_show_back_face=True, mesh_show_wireframe=False)

    if args.save:
        print('Store mesh at: ', mesh_out_file, 'Hooray!')
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