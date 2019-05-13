#!/bin/bash
# Setup data for ABX phone discrimination from shennong features on
# ZRC2015 datasets.
#
# This script prepares the data from raw Buckeye and Xitsonga
# distributions, compute the ABX tasks and extract the features

here=$(readlink -f $(dirname $0))
data_dir=$here/data

log_dir=$data_dir/log
mkdir -p $log_dir

# number of parallel jobs for features extraction
njobs=10

# cluster partition to schedule the jobs on
partition=all

echo "step 1: setup $data_dir"

module load anaconda/3
source activate shennong
$here/scripts/setup_data.py \
    $data_dir \
    /scratch1/data/raw_data/BUCKEYE/ \
    /scratch1/data/raw_data/NCHLT/nchlt_Xitsonga/ || exit 1

echo "step 2: setup abx tasks"

for corpus in english xitsonga
do
    item=$data_dir/$corpus.item
    for kind in across within
    do
        task=$data_dir/${corpus}_$kind.abx
        if [ $kind == within ]
        then
            options="-o phone -b talker context"
        else
            options="-o phone -a talker -b context"
        fi

        log=$log_dir/${corpus}_task_$kind.log
        rm -f $log

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${corpus}_${kind}
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1

module load anaconda/3
source activate abx

abx-task $item $task $options || exit 1
EOF
    done
done

echo "step 3: extracting features"

for config in $(find $data_dir/config -type f -name "*.yaml")
do
    for corpus in english xitsonga
    do
        log=$log_dir/${corpus}_$(basename $config .yaml).log
        rm -f $log

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$(basename $config |cut -d_ -f1)
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$njobs

module load anaconda/3
source activate shennong
export OMP_NUM_THREADS=1

$here/scripts/extract_features.py $data_dir $config $corpus --njobs $njobs || exit 1

EOF
    done
done


exit 0
