sequences=("../cfg/fusion_portable/canteen.yaml" "../cfg/fusion_portable/garden.yaml" \
           "../cfg/fusion_portable/mcr.yaml" "../cfg/newer_college/quad.yaml")

for s in ${sequences[@]}; do
  python3 run_rosbag.py $s \
    --overrides ../cfg/map_ablation.yaml --num_repeats 5 \
    --gpu_ids 0 1 --lite --run_all_combos
    
  python3 run_rosbag.py $s \
    --overrides ../cfg/traj_ablation.yaml  \
    --num_repeats 5 --gpu_ids 0 1 --lite
done