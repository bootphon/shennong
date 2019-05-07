#!/bin/bash

data_dir=$1
corpus=$2
kind=$3
njobs=$4

abx_dir=$data_dir/abx
mkdir -p $abx_dir

task=$data_dir/${corpus}_$kind.abx

for features in $(find $data_dir/features -type f -name "$corpus*.h5f")
do
    echo $features $corpus $kind
    base=$(basename $features .h5f)
    distance=$abx_dir/${kind}_$base.dist
    score=$abx_dir/${kind}_$base.score
    csv=$abx_dir/${kind}_$base.csv

    abx-distance -j $njobs -n 1 $features $task $distance || exit 1
    abx-score $task $distance $score || exit 1
    abx-analyze $score $task $csv || exit 1

    average=$($data_dir/../scripts/abx_average.py $csv $kind)
    [ -z $average ] && exit 1
    echo $(basename $csv .csv | tr -s '_' ' ') $average >> $data_dir/final_score.txt
done

exit 0
