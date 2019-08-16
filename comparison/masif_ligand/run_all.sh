#!/bin/bash
while read p
do
    ./run_probis.sh $p
done < benchmark_set.txt
