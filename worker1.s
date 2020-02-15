#!/bin/bash
#
#SBATCH --job-name=de1_c
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=5GB
#SBATCH --output=/scratch/ad5238/POET/UntouchableThunder/log/worker_%A_%a.out
#SBATCH --error=/scratch/ad5238/POET/UntouchableThunder/log/worker_%A_%a.err
#SBATCH --array=0-2
##SBATCH --mail-type=END
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=ad5238@nyu.edu

module purge
module load anaconda3/5.3.1
source activate thesis
module load gcc/6.3.0
module load jdk/11.0.4

sleep $(shuf -i 10-100 -n 1)

python child.py --id $SLURM_ARRAY_TASK_ID

#BLANK LINE UNDER THS LINE. SACRIFICE TO THE CARRIAGE RETURN GODS.
