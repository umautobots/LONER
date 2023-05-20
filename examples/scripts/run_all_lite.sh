sequences=("../cfg/fusion_portable/canteen.yaml" "../cfg/fusion_portable/garden.yaml" \
           "../cfg/fusion_portable/mcr.yaml" "../cfg/newer_college/quad.yaml" \
           "../cfg/newer_college/cloister.yaml" )

for s in ${sequences[@]}; do
  python3 run_rosbag.py $s \
    --overrides ../cfg/ablation_study.yaml --num_repeats 5 \
    --gpu_ids 0 1 2 3 --lite --run_all_combos 
done