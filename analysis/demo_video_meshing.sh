
exp_dir=$1
cfg_dir=$2
ckpt_step=$3
meshing_step=$4
resolution=$5

# Find the file with the largest number in its name
ckpt_path=$exp_dir/checkpoints
largest_file=$(ls -v $ckpt_path | grep -E "_[0-9]+\.tar$" | tail -n 1)

# Extract the number from the file name using a regular expression
if [[ $largest_file =~ ^.*_([0-9]+).*$ ]]; then
    max_id=${BASH_REMATCH[1]}
    echo "The maximum ckpt id is: $max_id"
else
    echo "Cannot find the maximum ckpt id."
fi


for ((i=0; i<=$max_id/$ckpt_step; i++))
do
  ckpt_id=$(($i*$ckpt_step))  
  echo "RUN $" python meshing.py $exp_dir $cfg_dir --ckpt_id $ckpt_id --use_weights --level 0.05 --save --skip_step $meshing_step --resolution $resolution
  python meshing.py $exp_dir $cfg_dir --ckpt_id $ckpt_id --use_weights --level 0.05 --save --skip_step $meshing_step --resolution $resolution
done

python3 meshing.py $exp_dir $cfg_dir --use_weights --level 0.05 --save --skip_step $meshing_step --resolution $resolution

