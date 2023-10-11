import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


q_lc = np.array([0.5, 0.5, 0.5, 0.5])
t_lc = np.array([0,0,0])


R_lc = Rotation.from_quat(np.array(q_lc)).as_matrix()

T_lc = np.vstack((np.hstack((R_lc, t_lc.reshape(-1, 1))),[0,0,0,1]))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ply_path")
parser.add_argument("resolution", type=float)
args = parser.parse_args()

print(f"Reading mesh from {args.ply_path}")
ply = o3d.io.read_triangle_mesh(args.ply_path)
cloud = ply.sample_points_uniformly(number_of_points=50_000_000)
cloud = cloud.voxel_down_sample(args.resolution)
print(f"Downsampled from 50million to {len(cloud.points)}.")
print(f"Saving to {args.ply_path[:-4]}_sampled.pcd")

o3d.io.write_point_cloud(f"{args.ply_path[:-4]}_sampled.pcd", cloud)


