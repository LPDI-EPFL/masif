# For the first of the two proteins, write a list of the residues that are to be blocked: those that are not in the target site. The target
# site, which would be the 'center of the interface'  is the patch with the highest shape complementarity to the target. 
#  utils/compute_blocked_res_list.py 

# Now run ZDock's create.pl to actually block these residues.

masif_root=$(git rev-parse --show-toplevel)
masif_src=$masif_root/source/
export PYTHON_PATH=$masif_src/:
while read pdbid_chain1_chain2
do
	echo $pdbid_chain1_chain2
	python3 util/compute_blocked_res_list.py $pdbid_chain1_chain2
done < ../benchmark_list.txt

