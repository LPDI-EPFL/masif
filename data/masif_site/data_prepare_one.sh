masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export masif_matlab
export PYTHONPATH=$PYTHONPATH:$masif_source
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
#python $masif_source/data_preparation/00-pdb_download.py $1
#python $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
#python $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
python $masif_source/data_preparation/02-compute_matlab_matrix.py $1
python $masif_source/data_preparation/03-compute_coords.py $1
python $masif_source/data_preparation/04-masif_precompute.py masif_site $1
