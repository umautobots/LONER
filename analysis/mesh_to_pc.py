import argparse
import os
import pathlib
import open3d as o3d
import numpy as np
import pickle
from src.common.pose_utils import WorldCube

parser = argparse.ArgumentParser("Sample Mesh to Pointcloud")
parser.add_argument("mesh_path", type=str, help="mesh")
# parser.add_argument("mesh_path", type=str, help="mesh")

# parser.add_argument("--ckpt_id", type=str, default=None)
# parser.add_argument("--resolution", type=float, default=0.1)
parser.add_argument("--viz", default=False, dest="viz", action="store_true")
args = parser.parse_args()

mesh_o3d = o3d.io.read_triangle_mesh(f"{args.mesh_path}")
# mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.zeros_like(mesh_o3d.vertex_colors))
mesh_o3d.compute_vertex_normals()
pcd = mesh_o3d.sample_points_uniformly(number_of_points=100000)

o3d.visualization.draw_geometries([mesh_o3d, pcd],
                                  mesh_show_back_face=True, mesh_show_wireframe=False)
o3d.visualization.draw_geometries([pcd],
                                  mesh_show_back_face=True, mesh_show_wireframe=False)