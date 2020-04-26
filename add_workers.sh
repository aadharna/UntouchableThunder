

#hello

for((i=31; i<35; i++)); do
	    OMP_NUM_THREADS=1 python -u child.py --id $i --exp_name $1 --args_file $2 > ./../results_$1/logs/w$i.log 2>&1 &
	    sleep 1
done
