sh nameone.sh "$1"

PAR=$(sbatch run.s $1)
sbatch --dependency=after:${PAR##* } worker1.s
sbatch --dependency=after:${PAR##* } worker2.s
sbatch --dependency=after:${PAR##* } worker3.s
sbatch --dependency=after:${PAR##* } worker4.s
sbatch --dependency=after:${PAR##* } worker5.s

