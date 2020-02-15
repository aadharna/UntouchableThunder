#!/bin/bash
#
#SBATCH --job-name=de1_p
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=20GB
#SBATCH --output=/scratch/ad5238/POET/UntouchableThunder/log/core_%A.out
##SBATCH --mail-type=END
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=ad5238@nyu.edu

# module purge
# module load anaconda3/5.3.1 
# source activate hpcthesis
# module load gcc/6.3.0
# module load jdk/11.0.4

arg=test

export OMP_NUM_THREADS=1

python poet_distributed.py --exp_name $arg > log/master.log 2>&1 &

sleep 60

python child.py --id 1  > log/1.log 2>&1 &
sleep 5
python child.py --id 2  > log/2.log 2>&1 &
sleep 5
python child.py --id 3  > log/3.log 2>&1 &
sleep 5


wait


# results directory placement is determined in args.yml
# on the HPC it is /scratch/ad5238/POET

#BLANK LINE UNDER THS LINE. SACRIFICE TO THE CARRIAGE RETURN GODS.

