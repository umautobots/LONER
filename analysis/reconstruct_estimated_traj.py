import pandas as pd
import argparse
import os, sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.common.pose_utils import build_poses_from_df, dump_trajectory_to_tum

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("keyframe_trajectory")
    parser.add_argument("tracked_trajectory")
    parser.add_argument("output_file")
    args = parser.parse_args()


    kf_df = pd.read_csv(args.keyframe_trajectory, delimiter=' ', header=None)
    tracked_df = pd.read_csv(args.tracked_trajectory, delimiter=' ', header=None)

    kf_traj, kf_times = build_poses_from_df(kf_df)
    tracked_traj, tracked_times = build_poses_from_df(tracked_df)

    # Entry (i,j) indicates whether frame i and keyframe j are at the same time
    kf_frame_indices = torch.where(tracked_times[:, None] == kf_times)[0]

    assert len(kf_frame_indices) == len(kf_times)

    reconstructed_traj = []

    for pose_idx, pose in enumerate(tracked_traj):
        reference_kf_idx = torch.argmin((kf_frame_indices <= pose_idx).float()) - 1
        reference_kf_pose = kf_traj[reference_kf_idx]

        reference_frame_idx = kf_frame_indices[reference_kf_idx]
        reference_frame_pose = tracked_traj[reference_frame_idx]

        T_ref_p = reference_frame_pose.inverse() @ pose
        opt_pose = reference_kf_pose @ T_ref_p
        reconstructed_traj.append(opt_pose)

    reconstructed_traj = torch.stack(reconstructed_traj)

    dump_trajectory_to_tum(reconstructed_traj, tracked_times, args.output_file)
