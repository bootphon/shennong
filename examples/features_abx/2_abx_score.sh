#!/bin/bash

here=$(readlink -f $(dirname $0))
data_dir=$here/data

log_dir=$data_dir/log
mkdir -p $log_dir

abx_dir=$data_dir/abx
mkdir -p $abx_dir

njobs=4

for features in $(find $data_dir/features -type f -name "*.h5f")
do
    corpus=$(basename $features | cut -d_ -f1)
    for kind in across within
    do
        task=$data_dir/${corpus}_$kind.abx
        distance=$abx_dir/${kind}_$(basename $features .h5f).dist
        score=$abx_dir/${kind}_$(basename $features .h5f).score
        csv=$abx_dir/${kind}_$(basename $features .h5f).csv
        average=$abx_dir/${kind}_$(basename $features .h5f).average

        log=$log_dir/${kind}_$(basename $features .h5f).log
        rm -f $log

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=abx
#SBATCH --output=$log
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$njobs


module load anaconda/3
source activate abx

abx-distance -j $njobs -n 1 $features $task $distance || exit 1
abx-score $task $distance $score || exit 1
abx-analyze $score $task $csv || exit 1
$here/scripts/abx_average.py $score $kind $average
EOF
    done
done

exit 0
