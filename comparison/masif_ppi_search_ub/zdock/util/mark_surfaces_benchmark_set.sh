# Generate the pdbs in the format that zdock needs them.
masif_root=$(git rev-parse --show-toplevel)
# The route to your zdock directory
#zdock_dir=$masif_root/ext_programs/zdock3.0.2_mac_intel/
zdock_dir=./
# The benchmark list
benchmark_list=$masif_root/comparison/masif_ppi_search_ub/benchmark_list.txt
# Location of source pdbs. 
PDBLOC=$masif_root/comparison/masif_ppi_search_ub/patchdock/pdbs/

mkdir -p output_tmp/

OUTDIR='01-zdock_marked/'
mkdir -p $OUTDIR

while read PDBID_CHAIN1_CHAIN2;
do
  PDBID=$(echo $PDBID_CHAIN1_CHAIN2| cut -d"_" -f1)
  CHAIN1=$(echo $PDBID_CHAIN1_CHAIN2| cut -d"_" -f2)
  CHAIN2=$(echo $PDBID_CHAIN1_CHAIN2| cut -d"_" -f3)

  PDB1=$PDBID\_$CHAIN1\.pdb
  # Remove all hydrogens, as reduce hydrogens are not recognized by marksur. 
  reduce -Trim $PDBLOC/$PDB1 > output_tmp/$PDB1
  # Mark the surface residues for zdock, save in temporary dir
  echo $zdock_dir/mark_sur output_tmp/$PDB1 $OUTDIR/$PDBID\_$CHAIN1\_m.pdb

  PDB2=$PDBID\_$CHAIN2\.pdb
#  # Remove all hydrogens, as reduce hydrogens are not recognized by marksur. 
  reduce -Trim $PDBLOC/$PDB2 > output_tmp/$PDB2
#  # Mark the surface residues for zdock, save in temporary dir
  $zdock_dir/mark_sur output_tmp/$PDB2 $OUTDIR/$PDBID\_$CHAIN2\_m.pdb

done < $benchmark_list

#rm -rf output_tmp
