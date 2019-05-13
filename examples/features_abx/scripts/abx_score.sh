#!/bin/bash

data_dir=$1
corpus=$2
task_type=$3
njobs=$4

abx_dir=$data_dir/abx
mkdir -p $abx_dir

task=$data_dir/${corpus}_$task_type.abx

for features in $(find $data_dir/features -type f -name "${corpus}_rasta*.h5f")
do
    echo $features $corpus $task_type
    base=$(basename $features .h5f)
    distance=$abx_dir/${task_type}_$base.dist
    score=$abx_dir/${task_type}_$base.score
    csv=$abx_dir/${task_type}_$base.csv

    abx-distance -j $njobs -n 1 $features $task $distance || exit 1
    abx-score $task $distance $score || exit 1
    abx-analyze $score $task $csv || exit 1

    average=$($data_dir/../scripts/abx_average.py $csv $task_type)
    [ -z $average ] && exit 1
    echo $(basename $csv .csv | tr -s '_' ' ') $average >> $data_dir/final_score.txt
done

exit 0
