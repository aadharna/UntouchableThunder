

sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker1.s
sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker2.s
sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker3.s
sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker4.s
sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_c/" worker5.s


sed -i.bak "3 s/^.*$/#SBATCH --job-name=$1_p/" run.s

