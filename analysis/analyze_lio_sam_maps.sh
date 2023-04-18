# Canteen
# for t in {0..4}; do
#   python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/canteen \
#     ~/data/fusion_portable/20220216_canteen_day/comparison_map.pcd \
#     --estimated_map trial_$t.pcd \
#     --gt_trajectory ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt \
#     --est_traj /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/canteen/trial_$t.tum \
#     --voxel_size 0.05
# done

# Garden
# for t in {0..4}; do
#   python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/garden \
#     ~/data/fusion_portable/20220216_garden_day/comparison_map.pcd \
#     --estimated_map trial_$t.pcd \
#     --gt_trajectory ~/data/fusion_portable/20220216_garden_day/ground_truth_traj.txt \
#     --est_traj /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/garden/trial_$t.tum \
#     --voxel_size 0.05
# done

# Quad
# for t in {0..4}; do
#   python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/quad \
#     ~/data/newer_college/quad/comparison_map.pcd \
#     --estimated_map trial_$t.pcd \
#     --gt_trajectory ~/data/newer_college/quad/ground_truth_traj.txt \
#     --est_traj /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/quad/trial_$t.tum \
#     --voxel_size 0.05
# done


# MCR
for t in {0..4}; do
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/liosam/mcr \
    ~/data/fusion_portable/20220219_MCR_slow_01/comparison_map.pcd \
    --estimated_map trial_$t.pcd \
    --voxel_size 0.01 \
    --initial_transform -0.108062 -0.994083 -0.011007 -3.508731 \
                          0.994076 -0.108178 0.010534 3.288983 \
                          -0.011662 -0.009804 0.999884 -0.398134 \
                          0.000000 0.000000 0.000000 1.000000
done





