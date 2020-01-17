which python
n=2
for ((i=1;i<=n;i++)); do
    python child.py --id $i &
done

python parent.py &
