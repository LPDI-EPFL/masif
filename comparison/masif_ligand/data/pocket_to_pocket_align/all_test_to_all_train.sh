#for i in {1..489}
for i in {1..489}
do
	python3 ./pocket_to_pocket_align.py $i &
	sleep 1
done
