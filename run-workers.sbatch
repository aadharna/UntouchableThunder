#!/bin/bash

#SBATCH --job-name=worker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --output=/scratch/ad5238/POET-20200221/UntouchableThunder/log/worker-%A_%a.out

module purge
module load gcc/6.3.0
source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate thesis

if [ $# -eq 2 ]; then
    if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then
	echo "argument missing"
	exit
    fi
elif [ $# -eq 3 ]; then
    export SLURM_ARRAY_TASK_ID=$3
fi

sleep 60

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u child.py --id $SLURM_ARRAY_TASK_ID --exp_name $1 --args_file $2






