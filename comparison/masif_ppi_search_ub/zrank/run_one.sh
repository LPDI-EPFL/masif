pdbdir=../patchdock/pdbs/
zdock_out_base=../zdock/03-results
PPI_PAIR_ID=$1
outdir=results/$PPI_PAIR_ID
mkdir -p $outdir
# Number of decoys per docked protein
NUM_DECOYS=2000

# Copy the already protonated files from the PatchDock directory, renaming them to ZDock's naming scheme.
PDBID=$(echo $PPI_PAIR_ID | cut -d"_" -f1) 
CHAIN1=$(echo $PPI_PAIR_ID | cut -d"_" -f2) 
CHAIN2=$(echo $PPI_PAIR_ID | cut -d"_" -f3) 

# Copy the target PDB file
cp $pdbdir/$PDBID\_$CHAIN1\.pdb $outdir/$PDBID\_$CHAIN1\_m_bl.pdb

# Copy all the 'decoy' pdbs, which will be needed to rank decoy complexes.
while read p
do
	DECOY_PDBID=$(echo $p | cut -d"_" -f1)
	DECOY_CHAIN2=$(echo $p | cut -d"_" -f3)
	cp $pdbdir/$DECOY_PDBID\_$DECOY_CHAIN2\.pdb $outdir/$DECOY_PDBID\_$DECOY_CHAIN2\_m.pdb
	cp ../zdock/03-results/$PPI_PAIR_ID/zdock_$PDBID\_$CHAIN1\_$DECOY_PDBID\_$DECOY_CHAIN2\.out $outdir/zdock_$PDBID\_$CHAIN1\_$DECOY_PDBID\_$DECOY_CHAIN2\.data
done < ../benchmark_list.txt

# Copy each zdock output file

# Copy create to directory.
cp zrank2_linux/zrank $outdir
cp ../zdock/create_lig $outdir
cp ../zdock/create.pl $outdir
cp eval_zrank.py $outdir

cd $outdir

# Edit the Zdock output file for the true pair to point to the current directory.

# Run ZRank2 on every pair.
for zdockout in zdock_*.data
do
	echo "Running ZRank on $zdockout"
	sed -i 's#../../01-zdock_marked/##' $zdockout
	/usr/bin/time -f 'user %U' -o $zdockout\_cpu_seconds.txt  ./zrank $zdockout 1 $NUM_DECOYS
	# Erase zdock output as it is no longer needed. 
done

# For the true pair, generate all complexes. 
./create.pl zdock_$PDBID\_$CHAIN1\_$PDBID\_$CHAIN2.data $NUM_DECOYS
#rm *.data

# Parse the top complexes and the zrank results.
python3 eval_zrank.py $PPI_PAIR_ID $PDBID\_$CHAIN1\_m_bl.pdb $PDBID\_$CHAIN2\_m.pdb $NUM_DECOYS > results.txt
# Erase all PDBs
#rm *.pdb
