masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
# Activate your environment here.
python -W ignore $masif_source/masif_ppi_search/second_stage_alignment_nn.py ../../../data/masif_ppi_search $1 2000 1000 masif
