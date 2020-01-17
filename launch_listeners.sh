which python
n=1
for ((i=1;i<=n;i++)); do
    python child.py --id $i
done

python parent.py
