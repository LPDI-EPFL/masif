#!/bin/sh
while read p
do
	./run_one.sh $p &
	sleep 1
done < ../benchmark_list.txt
