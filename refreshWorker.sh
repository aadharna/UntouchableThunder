


expname=$1
fname=$2
i=$3

echo "refreshingWorker $i"

OMP_NUM_THREADS=1 python -u child.py --id $i --exp_name $expname --args_file $fname >> ./../results_$1/logs/w$i.log 2>&1 &


