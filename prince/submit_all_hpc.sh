#!/bin/bash

if [ $# -eq 0 ]; then
    echo "argument missing"
    exit
fi

#sh nameone.sh "$1"
rm -rf ./../results_$1
mkdir -p ./../results_$1
cp ./args.yml ./../results_$1/

args=./../results_$1/args.yml

jobID=$(sbatch run-master.sbatch $1 $args | awk '{print $NF}')

sbatch --dependency=after:$jobID --array=1-11 run-workers.sbatch $1 $args

exit

