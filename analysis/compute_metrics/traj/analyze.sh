NUM_TRIAL=5

REL_DIST=3 #(m)


run_analysis() {
  mkdir -p ./$1/results/$2

  for ((i=0; i < NUM_TRIAL; i++)); do
    evo_ape tum $1/stamped_groundtruth.txt ./$1/$2/stamped_traj_estimate${i}.txt --t_max_diff 0.1 -a  --no_warnings --save_results ./$1/results/$2/trial_ape_$i.zip;
  done
  evo_res $1/results/$2/trial_ape_*.zip  --no_warnings --save_table ./$1/results/$2/statistics_ape.csv
  for ((i=0; i < NUM_TRIAL; i++)); do
    evo_rpe tum $1/stamped_groundtruth.txt ./$1/$2/stamped_traj_estimate${i}.txt --t_max_diff 0.1 -a -d $REL_DIST -u m --no_warnings --save_results ./$1/results/$2/trial_rpe_$i.zip;
  done
  evo_res $1/results/$2/trial_rpe_*.zip  --no_warnings --save_table ./$1/results/$2/statistics_rpe.csv

  for ((i=0; i < NUM_TRIAL; i++)); do
    evo_rpe tum $1/stamped_groundtruth.txt ./$1/$2/stamped_traj_estimate${i}.txt --t_max_diff 0.1 -a -d $REL_DIST -u m --no_warnings -r angle_deg --save_results ./$1/results/$2/trial_rre_$i.zip;
  done
  evo_res $1/results/$2/trial_rre_*.zip  --no_warnings --save_table ./$1/results/$2/statistics_rre.csv

  for ((i=0; i < NUM_TRIAL; i++)); do
    evo_ape tum $1/stamped_groundtruth.txt ./$1/$2/stamped_traj_estimate${i}.txt --t_max_diff 0.1 -a --no_warnings -r angle_deg --save_results ./$1/results/$2/trial_are_$i.zip;
  done
  evo_res $1/results/$2/trial_are_*.zip  --no_warnings --save_table ./$1/results/$2/statistics_are.csv
}

for f in $(ls); do
  for p in $(ls $f); do
    if [[ "$(basename $p)" == "results" || "$(basename $p)" == "stamped_groundtruth.txt" ]]; then
      continue
    fi

    if [[ ! -d $f/$p ]]; then
      continue
    fi

    run_analysis $f $p
    run_analysis $f $p
    run_analysis $f $p
  done
done
