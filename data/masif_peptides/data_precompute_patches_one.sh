masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_seed_root
export masif_source
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
source ~/lpdi_fs/masif/tensorflow-1.12_on_cpu/bin/activate
python $masif_source/data_preparation/04-masif_precompute.py masif_site $1
python $masif_source/data_preparation/04-masif_precompute.py masif_ppi_search $1
