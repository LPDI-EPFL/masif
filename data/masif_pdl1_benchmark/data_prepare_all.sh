#!/bin/bash

i=1
while read p; do
    ./data_prepare_one.sh $p
    i=$((i+1))
done < lists/masif_site_only.txt
