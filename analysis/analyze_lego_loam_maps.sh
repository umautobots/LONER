# Canteen
for t in {0..4}; do
  let j=$t+1
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/lego_loam/canteen/lego_loam \
    ~/data/fusion_portable/20220216_canteen_day/comparison_map.pcd \
    --estimated_map trial_$t.pcd \
    --gt_trajectory ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt \
    --est_traj /mnt/ws-frb/users//frank/frank/lego_loam/groundscan_id2/fusionportable/cateen/trial$j/res.tum \
    --voxel_size 0.05
done

# Garden
for t in {0..4}; do
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/lego_loam/garden/lego_loam \
    ~/data/fusion_portable/20220216_garden_day/comparison_map.pcd \
    --estimated_map trial_$t.pcd \
    --gt_trajectory ~/data/fusion_portable/20220216_garden_day/ground_truth_traj.txt \
    --est_traj /mnt/ws-frb/users//frank/frank/lego_loam/groundscan_id2/fusionportable/garden/trial$j/res.tum \
    --voxel_size 0.05
done

# Quad
for t in {0..4}; do
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/lego_loam/quad/lego_loam \
    ~/data/newer_college/quad/comparison_map.pcd \
    --estimated_map trial_$t.pcd \
    --gt_trajectory ~/data/newer_college/quad/ground_truth_traj.txt \
    --est_traj /mnt/ws-frb/users//frank/frank/lego_loam/default/newer_college/quad-easy/trial$j/res.tum \
    --voxel_size 0.05
done


# MCR
for t in {0..4}; do
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/lego_loam/mcr/lego_loam \
    ~/data/fusion_portable/20220219_MCR_slow_01/comparison_map.pcd \
    --estimated_map trial_$t.pcd \
    --voxel_size 0.01 \
    --initial_transform -0.994494 -0.006652 -0.104580 -3.563383 \
                        -0.104604 0.003451 0.994508 3.300708 \
                        -0.006254 0.999972 -0.004127 -0.405963 \
                        0.000000 0.000000 0.000000 1.000000
done





