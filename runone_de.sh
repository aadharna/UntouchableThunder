sh nameone_de.sh "$1"

mkdir ./../results_$1
cp ./args.yml ./../results_$1/

# clean out the .gradle folders and 
# rebuild gradle.
sh clean.sh

PAR=$(sbatch run_de.s $1)
sbatch $partitions --dependency=after:${PAR##* } worker1.s


