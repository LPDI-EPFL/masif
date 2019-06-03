#!/bin/bash
for pdb_id in 4ZQK_A 5JDS_A
do
    sbatch run_all.sh params $pdb_id
    sbatch run_all.sh params_small_proteins $pdb_id
done
