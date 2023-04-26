#!/usr/bin/env python
# coding: utf-8
import argparse
import pathlib
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")
import pickle
import torch
import re
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from src.common.pose import Pose
import rosbag
from mesher import Mesher
from analysis.utils import *
from scipy.spatial.transform import Rotation as R
import tqdm

class TUMPose:
    def __init__(self, timestamp, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def to_transform(self):
        r = R.from_quat([self.qx, self.qy, self.qz, self.qw])
        r_mat = r.as_matrix()
        t = np.array([self.x, self.y, self.z])
        T = np.eye(4)
        T[:3,:3] = r_mat
        T[:3,3] = t
        return T

def find_pose_by_timestamp(poses, timestamp):
    for pose in poses:
        if pose.timestamp == timestamp:
            return pose
    raise ValueError(f"No pose found with timestamp {timestamp}")

def find_closest_pose_by_timestamp(poses, timestamp):
    closest_pose = None
    closest_ts = None
    closest_dist = float('inf')
    for pose in poses:
        dist = abs(pose.timestamp - timestamp)
        if dist < closest_dist:
            closest_pose = pose
            closest_dist = dist
            closest_ts = pose.timestamp
    return closest_pose



def load_tum_trajectory(filename):
    poses = []
    with open(os.path.expanduser(filename), 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = line.strip().split(' ')
            timestamp = float(values[0])
            x, y, z = [float(v) for v in values[1:4]]
            qx, qy, qz, qw = [float(v) for v in values[4:]]
            pose = TUMPose(timestamp, x, y, z, qx, qy, qz, qw)
            poses.append(pose)
    return poses

def tumposes_ts_to_list(tumposes):
    ts_to_index_ = []
    for tumpose in tumposes:
        ts_to_index_.append(tumpose.timestamp)
    return ts_to_index_
def ckptposes_ts_to_list(ckptposes):
    ts_to_index_ = []
    for ckptpose in ckptposes:
        ts_to_index_.append(ckptpose['timestamp'])
    return ts_to_index_


def poses2lineset(lidar_pose_list: list, color=[0, 0, 0]):
    lines = []
    for i in range(len(lidar_pose_list)-1):
        lines.append([i, i+1])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([pose[:3, 3] for pose in lidar_pose_list]),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color(color)
    return line_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
    parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
    parser.add_argument("--use_gt_poses", default=False, dest="use_gt_poses", action="store_true")
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--viz", default=False, dest="viz", action="store_true")
    parser.add_argument("--use_weights", default=False, action="store_true")
    parser.add_argument("--level", type=float, default=0)
    parser.add_argument("--video_skip_step", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold for sigma MLP output. default as 0")
    parser.add_argument("--BEV", default=False, dest="bev", action="store_true")
    parser.add_argument("--smooth", default=False, dest="smooth", action="store_true")
    parser.add_argument("--only_last_frame", default=False, dest="only_last_frame", action="store_true")


    # parser.add_argument("--save", default=False, dest="save", action="store_true")
    args = parser.parse_args()
    checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")

    # load check points
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
    checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")
    ckpt = torch.load(str(checkpoint_path)) 

    # setup saving directory
    dir = args.experiment_directory

    suffix=''
    if args.bev:
        suffix+='_BEV'
    if args.smooth:
        suffix+='_smooth'

    render_img_path = f"{dir}/demo_video/image/"
    render_img_color_path = f"{dir}/demo_video/image_color/"
    render_depth_path = f"{dir}/demo_video/depth/"

    os.makedirs(render_img_path, exist_ok=True)
    os.makedirs(render_img_color_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)

    # override any params loaded from yaml
    with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)

    print('loading bag...')
    rosbag_path = full_config.dataset_path
    lidar_topic = '/'+full_config.system.ros_names.lidar
    bag = rosbag.Bag(rosbag_path, 'r')
    print('done loading bag')
    first_lidar_time = first_lidar_ts(bag, lidar_topic)

    gt_traj_file = full_config.run_config.groundtruth_traj
    gt_tumposes = load_tum_trajectory(gt_traj_file)
    gt_ts_list= tumposes_ts_to_list(gt_tumposes)
    est_traj_file = f'{args.experiment_directory}/trajectory/estimated_trajectory.txt'
    est_tumposes = load_tum_trajectory(est_traj_file)
    est_ts_list = tumposes_ts_to_list(est_tumposes)
    print(len(gt_ts_list))
    print(len(est_ts_list))

    # rendering camera config
    H = 1000
    W = 1000
    focal = 100 # 1
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

    r = R.from_euler('z', 45, degrees=True) # -25
    r = r.as_matrix()
    cam_pitch_down = np.identity(4)
    cam_pitch_down[:3,:3] = r

    lidar2cam = lidar2cam @ cam_pitch_down

    ckpt_poses = ckpt["poses"]
    ckptposes_ts_list = ckptposes_ts_to_list(ckpt_poses)

    map_o3d = o3d.geometry.PointCloud()
    lidar_pose_list=[]
    gt_lidar_pose_list=[]
    mesh = o3d.geometry.TriangleMesh()
    first_est_tumpose = None
    first_gt_tumpose = None

    for i, est_ts in tqdm.tqdm(enumerate(est_ts_list)):
        if i%args.video_skip_step!=0:
            continue
        if first_est_tumpose == None:
            first_est_tumpose = est_tumposes[i]
        if first_gt_tumpose == None:
            first_gt_tumpose = gt_tumposes[i]
        lidar_pose = np.linalg.inv(first_est_tumpose.to_transform()) @ est_tumposes[i].to_transform()
        lidar_pose_list.append(lidar_pose)

        gt_tum_pose = find_closest_pose_by_timestamp(gt_tumposes, first_lidar_time+est_ts)
        gt_lidar_pose = np.linalg.inv(first_gt_tumpose.to_transform()) @ gt_tum_pose.to_transform()
        gt_lidar_pose_list.append(gt_lidar_pose)

        # set camera view point
        c2w = lidar_pose @ lidar2cam
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array
        param.intrinsic = intrinsic
        ctr = vis.get_view_control()
        # ctr.set_constant_z_far(60)

        # store trajectory to a line_set
        est_line_set = poses2lineset(lidar_pose_list, [0, 0, 1])
        gt_line_set = poses2lineset(gt_lidar_pose_list, [1, 0, 0])
        
        # visualize current frame
        current_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.).transform(lidar_pose)
        
        if args.only_last_frame:
            if i!=len(est_ts_list)-1:
                continue
        
        print('load mesh')
        # get mesh
        ckpt_idx = np.sum(((np.array(ckptposes_ts_list)-est_ts)<0).astype(int))-1
        # mesh_file=f"{dir}/meshing/meshing_ckpt_{ckpt_idx}_res_{args.resolution}_weight_{args.use_weights}_level_{args.level}.ply"
        mesh_file=f"{dir}/meshing/resolution_{args.resolution}/ckpt_{ckpt_idx}.ply"
        if os.path.isfile(mesh_file):
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            if args.smooth:
                mesh = mesh.filter_smooth_simple(number_of_iterations=1) # smoothing for visualization
            mesh.compute_vertex_normals()
        else:
            print('cannot find' + mesh_file)
        
        # get corresponding lidar scan
        # lidar_msg = lidar_msg_list[np.argmin(np.abs(np.array(lidar_ts_list) - est_ts))]
        # lidar_o3d = o3d_pc_from_msg(lidar_msg).transform(lidar_pose)
        # map_o3d = merge_o3d_pc(map_o3d, lidar_o3d)

        # visualize in Open3D
        if args.viz:
            o3d.visualization.draw_geometries([mesh, current_frame, est_line_set, gt_line_set], mesh_show_back_face=True, mesh_show_wireframe=False)
            # o3d.visualization.draw_geometries([map_o3d])

        # render image
        vis.add_geometry(mesh, reset_bounding_box=True,)
        vis.add_geometry(current_frame, reset_bounding_box=True,)
        vis.add_geometry(est_line_set, reset_bounding_box=True,)
        vis.add_geometry(gt_line_set, reset_bounding_box=True,)
        # vis.add_geometry(lidar_o3d, reset_bounding_box=True,)

        if not args.bev:
            ctr.convert_from_pinhole_camera_parameters(param)
        else:
            param = o3d.camera.PinholeCameraParameters()
            # lidar2cam = np.array([[0,0,1,-1],
            #                     [-1,0,0,-1],
            #                     [0,-1,0,35],
            #                     [0,0,0,1]])
            # r = R.from_euler('x', -90, degrees=True)
            # r = r.as_matrix()
            # cam_pitch_down = np.identity(4)
            # cam_pitch_down[:3,:3] = r

            lidar2cam = np.array([[0,0,1,-6],
                                [-1,0,0,-2],
                                [0,-1,0,35],
                                [0,0,0,1]])
            r = R.from_euler('x', -90, degrees=True) # -25
            r = r.as_matrix()
            r_ = R.from_euler('z', -45, degrees=True)
            r = r @ r_.as_matrix()
            cam_pitch_down = np.identity(4)
            cam_pitch_down[:3,:3] = r

            c2w = lidar2cam @ cam_pitch_down
            param.extrinsic = np.linalg.inv(c2w) 
            param.intrinsic = intrinsic
            ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(True)
        # vis.get_render_option().mesh_shade_option= o3d.visualization.MeshShadeOption.Color
        vis.get_render_option().mesh_color_option= o3d.visualization.MeshColorOption.Color 
        image_color = vis.capture_screen_float_buffer(True)
        vis.get_render_option().mesh_color_option= o3d.visualization.MeshColorOption.Normal 
        image = vis.capture_screen_float_buffer(True)
        plt.imsave(f"{render_depth_path}"+"/{:05d}.png".format(i),np.asarray(depth), dpi = 1)
        plt.imsave(f"{render_img_path}"+"/{:05d}.png".format(i),np.asarray(image), dpi = 1)
        plt.imsave(f"{render_img_color_path}"+"/{:05d}.png".format(i),np.asarray(image_color), dpi = 1)
        vis.remove_geometry(mesh, reset_bounding_box=True,)
        vis.remove_geometry(current_frame, reset_bounding_box=True,)
        vis.remove_geometry(est_line_set, reset_bounding_box=True,)
        vis.remove_geometry(gt_line_set, reset_bounding_box=True,)
        # vis.remove_geometry(lidar_o3d, reset_bounding_box=True,)


