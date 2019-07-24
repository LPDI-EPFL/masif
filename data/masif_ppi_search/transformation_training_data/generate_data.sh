masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/:$masif_data/masif_ppi_search/transformation_training_data/
python3 $masif_source/masif_ppi_search/transformation_training_data/second_stage_transformation_training.py  $masif_data/masif_ppi_search/ 400 2000 9.0 ./transformation_data/ $1
