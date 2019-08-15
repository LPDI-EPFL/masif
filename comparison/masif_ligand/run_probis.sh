#!/bin/bash
cd data/training_srfs/
outid=$(echo $1| sed -e 's/.srf//g')
pdbid=$(echo $1 | cut -d"_" -f1)
chain=$(echo $1 | cut -d"_" -f2)
pdbid_chain=$pdbid\_$chain
../../probis/probis -results -z_score -1000 -noprune -longnames -ncpu 64 -surfdb -local -sfile training_srfs.txt -f1 ../../data/testing_srfs/$1 -c1 $chain -nosql ../../out_probis/sql/$outid\.sql
../../probis/probis -results -z_score -1000 -noprune -longnames -f1 ../../data/testing_pdbs/$pdbid_chain\.pdb -c1 $chain -nosql ../../out_probis/sql/$outid\.sql -json ../../out_probis/json/$outid\.json
