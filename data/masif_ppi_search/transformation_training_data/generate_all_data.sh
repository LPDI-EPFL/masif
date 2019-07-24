#for i in {1..1000}
for i in {10..1000}
do
	if ! ((i % 16)); then
		./generate_data.sh $i
	else
		./generate_data.sh $i &
	fi
done
