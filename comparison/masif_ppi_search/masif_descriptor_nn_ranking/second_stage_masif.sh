masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/:
python -u $masif_source/masif_ppi_search/nn_transform/second_stage_alignment_nn_score.py ../../../data/masif_ppi_search/ 3000 4000 100 masif
