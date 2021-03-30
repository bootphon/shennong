#!/bin/bash
# Run the whole experiment in one command, please customize the
# following parameters before launching it.

#####################
## parameters to tune

# path to the Buckeye corpus
buckeye_dir=/scratch1/data/raw_data/BUCKEYE/

# path to the NCHLT Xitsonga corpus
xitsonga_dir=/scratch1/data/raw_data/NCHLT/nchlt_Xitsonga/

# directory where to write all experiment data
data_dir=./data

# number of parallel jobs to launch
njobs=10

# cluster partition to schedule the jobs on
partition=all

# command to activate the shennong environment
activate_shennong="source /shared/apps/anaconda3/etc/profile.d/conda.sh; conda activate shennong"

# command to activate the abx environment
activate_abx="source /shared/apps/anaconda3/etc/profile.d/conda.sh; conda activate abx"

## end of parameters
####################


###############################
## do some checks on parameters

# check slurm is installed
if [ -z $(which sbatch 2> /dev/null) ]
then
    echo "error: slurm is not installed (sbatch not found)"
    exit 1
fi

# check data directory does not already exist
if [ -e $data_dir ]
then
    echo "error: $data_dir already exists"
    exit 1
fi

# check the corpora exist
for corpus in $buckeye_dir $xitsonga_dir
do
    if [ ! -d $corpus ]
    then
        echo "error: $corpus is not a directory"
        exit 1
    fi
done

## end of checks
################


# make the paths absolute
data_dir=$(readlink -f $data_dir)
buckeye_dir=$(readlink -f $buckeye_dir)
xitsonga_dir=$(readlink -f $xitsonga_dir)

# the directory where to find secondary scripts
scripts=$(readlink -f $(dirname $0))/scripts

# where to store log files
log_dir=$data_dir/log
mkdir -p $log_dir


echo "step 1: setup $data_dir"

eval $activate_shennong
$scripts/setup_data.py $data_dir $buckeye_dir $xitsonga_dir || exit 1


echo "step 2: setup abx tasks"

# create a temp file to store the script, erased at exit
step2=$(mktemp)
trap "rm -f $step2" EXIT

# prepare the dependency for step 3
dependency=afterok

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

        cat > $step2 <<EOF
#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1

$activate_abx
abx-task $item $task $options || exit 1
EOF

        pid=$(sbatch $step2 | cut -d' ' -f4)
        dependency=${dependency}:$pid
    done
done


echo "step 2 bis: extracting features"

for config in $(find $data_dir/config -type f -name "*.yaml")
do
    for corpus in english xitsonga
    do
        log=$log_dir/${corpus}_$(basename $config .yaml).log
        rm -f $log

        cat > $step2 <<EOF
#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$njobs

$activate_shennong
export OMP_NUM_THREADS=1

$scripts/extract_features.py $data_dir $config $corpus --njobs $njobs || exit 1
EOF

        pid=$(sbatch $step2 | cut -d' ' -f4)
        dependency=${dependency}:$pid
    done
done


echo "step 3: compute abx scores"

for corpus in english xitsonga
do
    for task_type in across within
    do
        log=$log_dir/abx_${corpus}_${task_type}.log
        rm -f $log

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=abx
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$njobs
#SBATCH --dependency=$dependency

$activate_abx
$scripts/abx_score.sh $data_dir $corpus $task_type $njobs || exit 1
EOF
    done
done

# collapse abx results
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=abx
#SBATCH --output=$log_dir/singleton_abx.log
#SBATCH --dependency=singleton

$activate_abx
$scripts/collapse_abx.py $data_dir -j $njobs
EOF

exit 0
