#!/bin/bash
#
#SBATCH --job-name=1_p
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --output=core_%A.out
#SBATCH --error=core_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ad5238@nyu.edu

module purge
module load anaconda3/5.3.1 gcc/6.3.0

source activate thesis

python poet_distributed.py --exp_name $1
# results directory placement is determined in args.yml
# on the HPC it is /scratch/ad5238/POET

#BLANK LINE UNDER THS LINE. SACRIFICE TO THE CARRIAGE RETURN GODS.
