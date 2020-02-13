

sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker1.s

sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_p/" run_de.s

