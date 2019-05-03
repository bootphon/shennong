#!/bin/bash

here=$(readlink -f $(dirname $0))

log_dir=$here/log
mkdir -p $log_dir

abx_dir=$here/data/abx
mkdir -p $abx_dir

for features in $(find $here/data/features -type f -name "*.h5f")
do
    corpus=$(basename $features | cut -d_ -f1)
    for kind in across within
    do
        task=$here/data/${corpus}_$kind.abx
        distance=$abx_dir/${kind}_$(basename $features .h5f).dist
        score=$abx_dir/${kind}_$(basename $features .h5f).score
        csv=$abx_dir/${kind}_$(basename $features .h5f).csv
        log=$log_dir/${kind}_$(basename $features .h5f).log
        rm -f $log

        sbatch -q all -n 10 -o $log <<EOF
#!/bin/bash
module load anaconda/3
source activate abx
abx-distance -j 10 -n 1 $features $task $distance
abx-score $task $distance $score
abx-analyze $score $task $csv
EOF
    done
done

exit 0
