#!/bin/bash
#
#SBATCH --job-name=20_c
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --output=worker_%A_%a.out
#SBATCH --error=worker_%A_%a.err
#SBATCH --array=10-19
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ad5238@nyu.edu

module purge
module load anaconda3/5.3.1 gcc/6.3.0

source activate thesis

python child.py --id $SLURM_ARRAY_TASK_ID

#BLANK LINE UNDER THS LINE. SACRIFICE TO THE CARRIAGE RETURN GODS.
