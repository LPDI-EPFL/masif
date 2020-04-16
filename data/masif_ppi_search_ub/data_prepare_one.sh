masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
# Invoke your environment here. 
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
python -W ignore $masif_source/data_preparation/04-masif_precompute.py masif_site $1
python -W ignore $masif_source/data_preparation/04-masif_precompute.py masif_ppi_search $1
