#!/bin/bash

here=$(readlink -f $(dirname $0))

log_dir=$here/log
mkdir -p $log_dir

for config in $(find $here/data/config -type f -name "*.yaml")
do
    for corpus in english xitsonga
    do
        log=$log_dir/${corpus}_$(basename $config .yaml).log
        rm -f $log
        sbatch -q all -n 10 -o $log <<EOF
#!/bin/bash
module load anaconda/3
source activate shennong
$here/scripts/extract_features.py $here/data $config $corpus --njobs 10
EOF
    done
done

exit 0
