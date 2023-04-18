# MCR
for t in {0..4}; do
  python3 evaluate_lidar_map.py /mnt/ws-frb/projects/loner_slam/baseline_map_evals_v2/nice_slam/mcr \
    ~/data/fusion_portable/20220219_MCR_slow_01/comparison_map.pcd \
    --estimated_map trial_$t\_sampled.pcd \
    --voxel_size 0.01 \
    --initial_transform 0.999195 -0.039322 -0.007921 -4.431115\
                        0.010072 0.054814 0.998446 2.935573\
                        -0.038827 -0.997722 0.055167 0.115403 \
                        0.000000 0.000000 0.000000 1.000000
done



