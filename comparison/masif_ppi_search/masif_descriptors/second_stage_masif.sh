masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
source /home/gainza/lpdi_fs/masif/tensorflow-1.12_on_cpu/bin/activate
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 100 4000 1000 masif 1
#python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 1000 4000 1000 masif
#python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 2000 4000 1000 masif
#python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search 3000 4000 1000 masif
