sh nameone.sh "$1"

mkdir ./../results_$1
cp ./args.yml ./../results_$1/

PAR=$(sbatch run.s $1)
sbatch $partitions --dependency=after:${PAR##* } worker1.s
#sbatch --dependency=after:${PAR##* } worker2.s
#sbatch --dependency=after:${PAR##* } worker3.s
#sbatch --dependency=after:${PAR##* } worker4.s
#sbatch --dependency=after:${PAR##* } worker5.s

