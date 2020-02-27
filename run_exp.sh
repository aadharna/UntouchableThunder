
arg=$1

mkdir -p ./../results_$1
mkdir -p ./../results_$1/logs
cp ./args.yml ./../results_$1/

source activate thesis; python -u poet_distributed.py --exp_name $arg > ./../results_$1/logs/master.log 2>&1 &

sleep 1

for((i=0; i<10; i++)); do
    source activate thesis; OMP_NUM_THREADS=1 python -u child.py --id $i > ./../results_$1/logs/w$i.log 2>&1 &
    sleep 2
done

wait
