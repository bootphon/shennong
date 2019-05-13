#!/bin/bash
# compute the ABX scores from features and tasks prepared in
# the script `1_setup_features.sh`

here=$(readlink -f $(dirname $0))
data_dir=$here/data

log_dir=$data_dir/log
mkdir -p $log_dir

njobs=10
partition=all

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

module load anaconda/3
source activate abx

$here/scripts/abx_score.sh $data_dir $corpus $task_type $njobs || exit 1

EOF
    done
done

exit 0
