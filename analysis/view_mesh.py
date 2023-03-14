import argparse
import pathlib
import os
import torch
import re
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from common.pose import Pose

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0,0,0])

parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
parser.add_argument("--ckpt_id", type=str, default=None)
parser.add_argument("--resolution", type=float, default=0.1)
parser.add_argument("--start_idx", type=int, default=0)

args = parser.parse_args()

# file_name = 'meshing_mcr_res0.02_smooth_taubin_it50'
# render_img_path = "/hostroot/home/pckung/TestData/"+file_name+"/image/"
# render_img_color_path = "/hostroot/home/pckung/TestData/"+file_name+"/image_color/"
# render_depth_path = "/hostroot/home/pckung/TestData/"+file_name+"/depth/"

# mesh = o3d.io.read_triangle_mesh(f"/hostroot/home/pckung/meshing_color_res0.05.ply")
# mesh = o3d.io.read_triangle_mesh("/hostroot/home/pckung/meshing_mcr_res0.02.ply")


render_img_path = f"{args.experiment_directory}/meshing/ckpt_{args.ckpt_id}/image/"
render_img_color_path = f"{args.experiment_directory}/meshing/ckpt_{args.ckpt_id}/image_color/"
render_depth_path = f"{args.experiment_directory}/meshing/ckpt_{args.ckpt_id}/depth/"
mesh = o3d.io.read_triangle_mesh(f"{args.experiment_directory}/meshing/meshing_ckpt_{args.ckpt_id}_res_{args.resolution}.ply")

# mesh = mesh.filter_smooth_simple(number_of_iterations=1)
mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
mesh.compute_vertex_normals()

checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")
if args.ckpt_id is None:
    #https://stackoverflow.com/a/2669120
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
elif args.ckpt_id=='final':
    checkpoint = f"final.tar"
else:
    checkpoint = f"ckpt_{args.ckpt_id}.tar"
checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")
if not checkpoint_path.exists():
    print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    exit()
print(f'Loading checkpoint from: {checkpoint_path}') 
ckpt = torch.load(str(checkpoint_path)) 

if not os.path.exists(render_img_path):
    os.makedirs(render_img_path)
if not os.path.exists(render_img_color_path):
    os.makedirs(render_img_color_path)
if not os.path.exists(render_depth_path):
    os.makedirs(render_depth_path)

H = 1000
W = 1000
focal = 10
fx = focal
fy = focal
cx = H/2.0-0.5
cy = W/2.0-0.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
vis = o3d.visualization.Visualizer()
vis.create_window(width=W, height=H)
vis.get_render_option().mesh_show_back_face = True
vis.get_render_option().mesh_color_option= o3d.visualization.MeshColorOption.Color # Normal # ZCoordinate

param = o3d.camera.PinholeCameraParameters()
lidar2cam = np.array([[0,0,1,0],
                      [-1,0,0,0],
                      [0,-1,0,0],
                      [0,0,0,1]])

poses = ckpt["poses"]
for i, kf in enumerate(poses):
    if i<args.start_idx:
        continue
    if args.use_gt_poses:
        pose_key = "gt_lidar_pose"
    else:
        pose_key = "lidar_pose"
    lidar_pose = Pose(pose_tensor=kf[pose_key]).get_transformation_matrix().cpu().numpy()
    c2w = lidar_pose @ lidar2cam

    param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array
    param.intrinsic = intrinsic
    ctr = vis.get_view_control()
    # ctr.set_constant_z_far(60)

    vis.add_geometry(mesh, reset_bounding_box=True,)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(True)
    vis.get_render_option().mesh_color_option= o3d.visualization.MeshColorOption.Color 
    image_color = vis.capture_screen_float_buffer(True)
    vis.get_render_option().mesh_color_option= o3d.visualization.MeshColorOption.Normal 
    image = vis.capture_screen_float_buffer(True)
    plt.imsave(f"{render_depth_path}"+"/{:05d}.png".format(i),np.asarray(depth), dpi = 1)
    plt.imsave(f"{render_img_color_path}"+"/{:05d}.png".format(i),np.asarray(image_color), dpi = 1)
    plt.imsave(f"{render_img_path}"+"/{:05d}.png".format(i),np.asarray(image), dpi = 1)
    vis.remove_geometry(mesh, reset_bounding_box=True,)

# o3d.visualization.draw_geometries([mesh, origin_frame], mesh_show_back_face=True, mesh_show_wireframe=False)


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# ctr = vis.get_view_control()
# pcd = o3d.io.read_point_cloud("/hostroot/mnt/ws-frb/projects/cloner_slam/fusion_portable/20220216_canteen_day/000000.pcd")
# vis.add_geometry(pcd)
# vis.run() # user changes the view and press "q" to terminate
# param = ctr.convert_to_pinhole_camera_parameters()
# trajectory = o3d.camera.PinholeCameraTrajectory()
# trajectory.intrinsic = param[0]
# trajectory.extrinsic = o3d.utility.Matrix4dVector([param[1]])
# o3d.camera.write_pinhole_camera_trajectory("test.json", trajectory)
# vis.destroy_window()