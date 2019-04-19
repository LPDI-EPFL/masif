masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 100 2000 100 gif
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 500 2000 100 gif
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 1000 2000 100 gif
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 3000 2000 100 gif
