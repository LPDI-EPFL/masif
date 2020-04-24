#!/bin/bash
cat ../../comparison/masif_ppi_search/benchmark_list.txt | while read line 
do
    ./data_prepare_one.sh $line
    ./compute_descriptors.sh $line
done

