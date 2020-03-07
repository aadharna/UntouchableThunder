
arg=$1

mkdir -p ./../results_$1
mkdir -p ./../results_$1/logs
cp ./args.yml ./../results_$1/

fname="./../results_$1/args.yml"

python -u poet_distributed.py --exp_name $arg --args_file $fname > ./../results_$1/logs/master.log 2>&1 &

sleep 1

for((i=0; i<10; i++)); do
    OMP_NUM_THREADS=1 python -u child.py --id $i --exp_name $arg --args_file $fname > ./../results_$1/logs/w$i.log 2>&1 &
    sleep 1
done

wait
