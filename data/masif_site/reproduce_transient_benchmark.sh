#!/bin/bash
cat lists/testing_transient.txt | while read line 
do
    ./data_prepare_one.sh $line\_
done

./predict_site.sh -l lists/testing_transient.txt
./color_site.sh -l lists/testing_transient.txt
