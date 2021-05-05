#!/bin/bash
#!/bin/bash
# Run the whole experiment in one command, please customize the following
# parameters before launching it. REquires a cluster running SLURM.

#####################
## parameters to tune

# path to the Buckeye corpus
buckeye_dir=/scratch1/data/raw_data/BUCKEYE/

# directory where to write all experiment data
data_dir=./data

# number of SLURM jobs to generate for VTLN training
njobs_vtln=30

# number of CPU cores to use for VTLN training
ncores=4

# # number of parallel jobs per task for features extraction and ABX evaluation
# njobs=10

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
if [ ! -d $buckeye_dir ]
then
    echo "error: $buckeye_dir is not a directory"
    exit 1
fi

## end of checks
################


# make the paths absolute
data_dir=$(readlink -f $data_dir)
buckeye_dir=$(readlink -f $buckeye_dir)

# the directory where to find secondary scripts
scripts=$(readlink -f $(dirname $0))/scripts

# where to store log files
log_dir=$data_dir/log
mkdir -p $log_dir

# create a temp file to store the script, erased at exit
tempfile=$(mktemp)
trap "rm -f $tempfile" EXIT

# prepare the dependency for step 4
dependency=afterok



echo "step 1: setup $data_dir"

eval $activate_shennong
$scripts/setup_data.py $data_dir $buckeye_dir -n $njobs_vtln || exit 1



echo "step 2: setup abx task"

log=$log_dir/abx_task.log
rm -f $log

cat > $tempfile <<EOF
#!/bin/bash
#SBATCH --job-name=task
#SBATCH --output=$log
#SBATCH --partition=$partition
#SBATCH --ntasks=1

export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

$activate_abx
abx-task $data_dir/english.item $data_dir/english_across.abx -o phone -a talker -b context || exit 1
EOF

pid=$(sbatch $tempfile | cut -d' ' -f4)
dependency=${dependency}:$pid



echo "step 3: compute VTLN warps"

rm -f $log_dir/vtln_*.log

cat > $tempfile <<EOF
#!/bin/bash
#SBATCH --job-name=vtln_%a
#SBATCH --output=${log_dir}/vtln_%a.log
#SBATCH --partition=$partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$ncores

export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

$activate_shennong
while read file
do
  $scripts/extract_warps.py $file --njobs $ncores
done < <(grep "^${SLURM_ARRAY_TASK_ID}" $data_dir/vtln_jobs.txt | cut -d" " -f2)
EOF

pid=$(sbatch --array=1-${njobs_vtln} $tempfile | cut -d' ' -f4)
dependency=${dependency}:$pid
