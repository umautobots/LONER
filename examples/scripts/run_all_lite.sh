sequences=("../cfg/fusion_portable/canteen.yaml" "../cfg/fusion_portable/garden.yaml" \
	           "../cfg/fusion_portable/mcr.yaml" "../cfg/newer_college/quad.yaml" \
		              "../cfg/newer_college/cloister.yaml" )

for s in ${sequences[@]}; do
	  python3 run_rosbag.py $s  --num_repeats 3 \
		          --gpu_ids 0 1 2 3 --lite
done
