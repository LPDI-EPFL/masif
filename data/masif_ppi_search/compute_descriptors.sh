masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/
#source $masif_source/tensorflow-1.9/bin/activate
python $masif_source/masif_ppi_search/masif_ppi_search_comp_desc.py $1 -l lists/full_list.txt
