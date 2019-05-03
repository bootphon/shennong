#!/bin/bash

here=$(readlink -f $(dirname $0))

log_dir=$here/log
mkdir -p $log_dir

for corpus in english xitsonga
do
    item=$here/data/$corpus.item
    for kind in across within
    do
        task=$here/data/${corpus}_$kind.abx
        [ $kind == within ] \
            && options="-o phone -a context -b talker" \
                || options="-o phone -a context talker"

        log=$log_dir/${corpus}_task_$kind.log
        sbatch -q all -n 1 -o $log <<EOF
#!/bin/bash
module load anaconda/3
source activate abx
abx-task $item $task $options
EOF
    done
done

exit 0
