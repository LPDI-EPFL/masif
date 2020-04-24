#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16384
#SBATCH --time 03:00:00
#SBATCH --array=1-1000
#SBATCH --output=exelogs/_masif_precompute.%A_%a.out
#SBATCH --error=exelogs/_masif_precompute.%A_%a.err

i=1
while read p; do
    if [ $(( i % 1001 )) == ${SLURM_ARRAY_TASK_ID} ]; then
        FIELD1=$(echo $p| cut -d" " -f1)
        PDBID=$(echo $FIELD1| cut -d"_" -f1)
        CHAIN1=$(echo $FIELD1| cut -d"_" -f2)
        ./data_extract_helix_one.sh $PDBID\_$CHAIN1
    fi
    i=$((i+1))
done < lists/bc-100-list.txt
