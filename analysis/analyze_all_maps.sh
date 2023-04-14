# MCR
for f in $(find /mnt/ws-frb/projects/loner_slam/evaluation_v3/lite -name "trial*" |grep mcr); do
  echo Analyzing $f
  python3 evaluate_lidar_map.py $f ~/data/fusion_portable/20220219_MCR_slow_01/comparison_map.pcd \
    --estimated_map meshing/meshing_ckpt_final_res_0.05_sampled.pcd \
    --voxel_size 0.01 \
    --initial_transform -0.106697 -0.994264 -0.007467 -3.488138 \
                         0.994281 -0.106727 0.003704 3.225835 \
                         -0.004480 -0.007029 0.999965 -0.434129 \
                         0.000000 0.000000 0.000000 1.000000
done


# Garden
for f in $(find /mnt/ws-frb/projects/loner_slam/evaluation_v3/ -name "trial*" |grep garden); do
  echo Analyzing $f
  python3 evaluate_lidar_map.py $f ~/data/fusion_portable/20220216_garden_day/comparison_map.pcd \
    --gt_trajectory ~/data/fusion_portable/20220216_garden_day/ground_truth_traj.txt \
    --est_traj $f/trajectory/estimated_trajectory.txt \
    --estimated_map meshing/meshing_ckpt_final_res_0.05_sampled.pcd \
    --voxel_size 0.05
done

# Canteen
for f in $(find /mnt/ws-frb/projects/loner_slam/evaluation_v3/ -name "trial*" |grep canteen); do
  echo Analyzing $f
  python3 evaluate_lidar_map.py $f ~/data/fusion_portable/20220216_canteen_day/comparison_map.pcd \
    --gt_trajectory ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt \
    --est_traj $f/trajectory/estimated_trajectory.txt \
    --estimated_map meshing/meshing_ckpt_final_res_0.05_sampled.pcd \
    --voxel_size 0.05
done

# Quad
for f in $(find /mnt/ws-frb/projects/loner_slam/evaluation_v3/ -name "trial*" |grep quad); do
  echo Analyzing $f
  python3 evaluate_lidar_map.py $f ~/data/newer_college/quad/comparison_map.pcd \
    --gt_trajectory ~/data/newer_college/quad/ground_truth_traj.txt \
    --est_traj $f/trajectory/estimated_trajectory.txt \
    --estimated_map meshing/meshing_ckpt_final_res_0.05_sampled.pcd \
    --voxel_size 0.05
done
