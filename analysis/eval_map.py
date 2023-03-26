import argparse
import random

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
import os
import pathlib
import re
import pickle
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")
from src.models.model_tcnn import Model, OccupancyGridModel
from src.common.pose_utils import WorldCube

canteen_tf = np.eye(4)
canteen_tf[:3,3] = np.array([14.519333839, 6.672959328, -0.265248626])
canteen_tf[:3,:3] = R.from_quat([-0.012090746, 0.023202606, -0.180399119, 0.983245448]).as_matrix()

mcr_tf = np.asarray([[-1.03415828e-01, -9.94606513e-01, -7.94047167e-03, -3.57574629e+00],
                [ 9.94637372e-01, -1.03422799e-01,  4.71267195e-04,  3.26781082e+00],
                [-1.28995122e-03, -7.84915339e-03,  9.99968363e-01, -4.04234611e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

nice_slam2cloner_tf = np.linalg.inv(np.asarray([[0., -1., -0.,  0.],
                                                [0.,  0., -1.,  0.],
                                                [1.,  0.,  0.,  0.],
                                                [0.,  0.,  0.,  1.]]))


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.5): # 0.05
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float64))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_pc, gt_pc):
    print('align')
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    trans_init = np.eye(4)
    threshold = 0.3
    reg_p2p = o3d.pipelines.registration.registration_icp(
        rec_pc, gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000, relative_fitness=1e-10, relative_rmse=1e-10))
    transformation = reg_p2p.transformation

    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        rec_pc, gt_pc, threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000, relative_fitness=1e-10, relative_rmse=1e-10))
    transformation = reg_p2p.transformation
    print('transformation:\n', transformation)
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_mesh_to_pc_metric(rec_meshfile, gt_pcfile, align=True):
    """
    3D reconstruction metric.

    """
    print('Loading mesh...')
    rec_mesh_o3d = o3d.io.read_triangle_mesh(f"{rec_meshfile}")
    rec_mesh_o3d.compute_vertex_normals()
    rec_mesh_sample_pc = o3d.geometry.PointCloud()
    rec_mesh_sample_pc.points = rec_mesh_o3d.vertices
    # rec_mesh_sample_pc = rec_mesh_o3d.sample_points_uniformly(number_of_points=100000)
    rec_mesh_sample_pc.paint_uniform_color(np.array([1, 0.1, 0]))

    print('Loading gt point cloud...')
    gt_pc = o3d.io.read_point_cloud(gt_pcfile)
    gt_pc = gt_pc.voxel_down_sample(0.1)
    gt_pc.paint_uniform_color(np.array([0, 0.1, 1]))

    print(np.asarray(gt_pc.points).shape)
    print(np.asarray(rec_mesh_sample_pc.points).shape)
    
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([rec_mesh_o3d, mesh_frame])

    rec_mesh_sample_pc = rec_mesh_sample_pc.transform(canteen_tf)
    rec_mesh_o3d = rec_mesh_o3d.transform(canteen_tf)
    # threshold data
    rec_mesh_sample_pc = rec_mesh_sample_pc.select_by_index(np.where(np.asarray(rec_mesh_sample_pc.points)[:, 2] < 10)[0])
    gt_pc = gt_pc.select_by_index(np.where(np.asarray(gt_pc.points)[:, 2] < 10)[0])


    # rec_mesh_sample_pc = rec_mesh_sample_pc.transform(nice_slam2cloner_tf)
    # rec_mesh_o3d = rec_mesh_o3d.transform(nice_slam2cloner_tf)
    # rec_mesh_sample_pc = rec_mesh_sample_pc.transform(mcr_tf)
    # rec_mesh_o3d = rec_mesh_o3d.transform(mcr_tf)

    rec_mesh_sample_pc = rec_mesh_sample_pc.voxel_down_sample(0.1)

    # if align:
    #     transformation = get_align_transformation(rec_mesh_sample_pc, gt_pc)
    #     rec_mesh_sample_pc = rec_mesh_sample_pc.transform(transformation)
    #     rec_mesh_o3d = rec_mesh_o3d.transform(transformation)

    lidar_map_gt = o3d.io.read_point_cloud('/hostroot/mnt/ws-frb/users/frank/frank/cloner-slam_experiment_data/mcr_occ_size1000_lr0.001_032123_180843/meshing/lidar_map_gt.pcd')
    lidar_map = o3d.io.read_point_cloud('/hostroot/mnt/ws-frb/users/frank/frank/cloner-slam_experiment_data/mcr_occ_size1000_lr0.001_032123_180843/meshing/lidar_map.pcd')
    lidar_map_gt = lidar_map_gt.transform(mcr_tf)
    lidar_map_gt = lidar_map_gt.voxel_down_sample(0.1)
    lidar_map = lidar_map.transform(mcr_tf)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([gt_pc, rec_mesh_o3d, mesh_frame],
                                  mesh_show_back_face=True, mesh_show_wireframe=False)
    
    # o3d.visualization.draw_geometries([gt_pc, lidar_map_gt, mesh_frame])
    
    o3d.visualization.draw_geometries([gt_pc, rec_mesh_sample_pc, mesh_frame])
    o3d.visualization.draw_geometries([gt_pc])
    o3d.visualization.draw_geometries([rec_mesh_sample_pc, mesh_frame])

    print('Claculating accuracy')
    accuracy_rec = accuracy(np.asarray(gt_pc.points), np.asarray(rec_mesh_sample_pc.points))
    print('accuracy_rec: ', accuracy_rec)

    completion_rec = completion(np.asarray(gt_pc.points), np.asarray(rec_mesh_sample_pc.points))
    print('completion_rec: ', completion_rec)

    completion_ratio_rec = completion_ratio(np.asarray(gt_pc.points), np.asarray(rec_mesh_sample_pc.points))
    print('completion_ratio_rec: ', completion_ratio_rec)

    # mesh_rec = trimesh.load(rec_meshfile, process=False)
    # mesh_gt = trimesh.load(gt_meshfile, process=False)

    # if align:
    #     transformation = get_align_transformation(rec_meshfile, gt_meshfile)
    #     mesh_rec = mesh_rec.apply_transform(transformation)

    # rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    # rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    # gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    # gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    # accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    # completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    # completion_ratio_rec = completion_ratio(
    #     gt_pc_tri.vertices, rec_pc_tri.vertices)
    # accuracy_rec *= 100  # convert to cm
    # completion_rec *= 100  # convert to cm
    # completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio: ', completion_ratio_rec)


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    unseen_gt_pointcloud_file = gt_meshfile.replace('.ply', '_pc_unseen.npy')
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in range(n_imgs):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target)-np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if (~seen):
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(gt_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(gt_mesh, reset_bounding_box=True,)

        vis.add_geometry(rec_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True,)

        errors += [np.abs(gt_depth-ours_depth).mean()]

    errors = np.array(errors)
    # from m to cm
    print('Depth L1: ', errors.mean()*100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the reconstruction.'
    )
    # parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
    # parser.add_argument("--ckpt_id", type=str, default=None)

    parser.add_argument('--rec_mesh', type=str,
                        help='reconstructed mesh file path')
    parser.add_argument('--gt_pc', type=str,
                        help='ground truth point cloud file path')
    parser.add_argument('-2d', '--metric_2d',
                        action='store_true', help='enable 2D metric')
    parser.add_argument('-3d', '--metric_3d',
                        action='store_true', help='enable 3D metric')
    args = parser.parse_args()

    # checkpoints = os.listdir(f"{args.experiment_directory}/checkpoints")
    # if args.ckpt_id is None:
    #     #https://stackoverflow.com/a/2669120
    #     convert = lambda text: int(text) if text.isdigit() else text 
    #     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    #     checkpoint = sorted(checkpoints, key = alphanum_key)[-1]
    #     args.ckpt_id = checkpoint.split('.')[0]
    # elif args.ckpt_id=='final':
    #     checkpoint = f"final.tar"
    # else:
    #     checkpoint = f"ckpt_{args.ckpt_id}.tar"
    # checkpoint_path = pathlib.Path(f"{args.experiment_directory}/checkpoints/{checkpoint}")
    
    # # override any params loaded from yaml
    # with open(f"{args.experiment_directory}/full_config.pkl", 'rb') as f:
    #     full_config = pickle.load(f)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # _DEVICE = torch.device(full_config.mapper.device)
    # print('_DEVICE', _DEVICE)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # if not checkpoint_path.exists():
    #     print(f'Checkpoint {checkpoint_path} does not exist. Quitting.')
    #     exit()

    # occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
    # assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
    # scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
    # shift = full_config.world_cube.shift
    # world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

    # # use single fine MLP when using OGM
    # model_config = full_config.mapper.optimizer.model_config.model
    # model = Model(model_config).to(_DEVICE)

    # print(f'Loading checkpoint from: {checkpoint_path}') 
    # ckpt = torch.load(str(checkpoint_path))

    # scale_factor = full_config.world_cube.scale_factor.cpu().numpy()
    # shift = full_config.world_cube.shift.cpu().numpy()
    # shift = np.concatenate((shift, np.array([0.]))).reshape((1,4))

    # occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
    # occ_model.load_state_dict(ckpt['occ_model_state_dict'])
    # occupancy_grid = occ_model()
    # occ_sigma_np = occupancy_grid.squeeze().cpu().detach().numpy()
    # if occ_sigma_np.sum() > 1:
    #     occ_probs = 1. / (1 + np.exp(-occ_sigma_np))
    #     occ_probs = (510 *  (occ_probs.clip(0.5, 1.0) - 0.5)).astype(np.uint8).reshape(-1)
    #     nonzero_indices = occ_probs.nonzero()
    #     x_ = np.arange(occ_model_config.voxel_size)
    #     x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    #     # X = np.stack([x.reshape(-1)[nonzero_indices], y.reshape(-1)[nonzero_indices], -z.reshape(-1)[nonzero_indices], occ_probs[nonzero_indices]], axis=1)
    #     X = np.stack([x.reshape(-1)[nonzero_indices], y.reshape(-1)[nonzero_indices], z.reshape(-1)[nonzero_indices], occ_probs[nonzero_indices]], axis=1)
    #     print('X.shape: ', X.shape)

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(X[:,:3]-(occ_model_config.voxel_size/2.))
    #     pcd.colors = o3d.utility.Vector3dVector(np.repeat(X[:,3].reshape((-1,1)), 3, axis=1)/255.)
    #     print(np.mean(X[:,:3], axis=0))
    #     print(np.mean(np.repeat(X[:,3].reshape((-1,1)), 3, axis=1), axis=0))
    #     print(np.max(X[:,3]), np.min(X[:,3]))
    #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd, mesh_frame])


    if args.metric_3d:
        calc_3d_mesh_to_pc_metric(args.rec_mesh, args.gt_pc)

    if args.metric_2d:
        calc_2d_metric(args.rec_mesh, args.gt_mesh, n_imgs=1000)
