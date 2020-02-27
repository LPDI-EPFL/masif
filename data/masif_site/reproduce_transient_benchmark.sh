#!/bin/bash
cat lists/testing_transient.txt | while read line 
do
   # do something with $line here
    ./data_prepare_one.sh $line\_
done
