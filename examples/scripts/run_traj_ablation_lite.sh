CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_rosbag.py $1 \
    --overrides ../cfg/traj_ablation.yaml  \
    --num_repeats 5 --gpu_ids 0 1 2 3 --lite