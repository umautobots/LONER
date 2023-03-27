import rosbag
import ros_numpy
import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

bag_name = '/hostroot/mnt/ws-frb/projects/cloner_slam/fusion_portable/20220219_MCR_slow_01/20220219_MCR_slow_01_ref_rgbd.bag'
lidar_topic = '/os_cloud_node/points'

gt_pcfile = '/hostroot/mnt/ws-frb/projects/cloner_slam/fusion_portable/groundtruth/map/20220216_MCR/merged_scan.pcd'
gt_pc = o3d.io.read_point_cloud(gt_pcfile)

bag = rosbag.Bag(bag_name, 'r')

trans_init = np.eye(4)
trans_init[:3,3] = np.array([0,0,0])
trans_init[:3,:3] = R.from_euler('z', 90, degrees=True).as_matrix()

def get_align_transformation(rec_pc, gt_pc, trans_init):
    print('align')
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    threshold = 2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        rec_pc, gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-10, relative_rmse=1e-10))
    transformation = reg_p2p.transformation

    threshold = 0.3
    reg_p2p = o3d.pipelines.registration.registration_icp(
        rec_pc, gt_pc, threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000, relative_fitness=1e-10, relative_rmse=1e-10))
    transformation = reg_p2p.transformation
    print('transformation:\n', transformation)
    return transformation

for topic, msg, ts in bag.read_messages(topics=[lidar_topic]):

    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    lidar_data = pd.DataFrame(lidar_data).to_numpy()
    xyz = lidar_data[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([gt_pc, pcd, mesh_frame])

    transformation = get_align_transformation(pcd, gt_pc, trans_init)
    pcd = pcd.transform(transformation)

    o3d.visualization.draw_geometries([gt_pc, pcd, mesh_frame])
    break
