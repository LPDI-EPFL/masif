masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
# Load your environment here. 
python $masif_source/masif_ppi_search/differentiable_alignment/make_feats.py $PDB_ID\_$CHAIN1\_$CHAIN2
