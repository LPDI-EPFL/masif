# Generate the pdbs in the format that zdock needs them.
masif_root=$(git rev-parse --show-toplevel)
# The route to your zdock directory
zdock_dir=$masif_root/ext_programs/zdock3.0.2_linux_x64/
# The benchmark list
benchmark_list=$masif_root/comparison/masif_ppi_search/benchmark_list.txt
# Mark the surface residues for zdock. 
$zdock_dir/mark_sur
