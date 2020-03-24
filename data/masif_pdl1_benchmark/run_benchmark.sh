masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:
python $masif_source/masif_ppi_search/pdl1_benchmark/pdl1_benchmark.py 
