PDBID_CHAIN1_CHAIN2=$1
PDBID=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f1)
CHAIN1=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f2)
CHAIN2=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f3)

outdir=03-results/$PDBID_CHAIN1_CHAIN2/
mkdir -p $outdir
# Copy target pdb, with surface marked and blocked residues.
cp 02-zdock_marked_blocked_pdbs/$PDBID\_$CHAIN1\_m_bl.pdb $outdir

# Copy ligand pdb of the co-crystal binder, with surface marked
cp 01-zdock_marked/$PDBID\_$CHAIN2\_m.pdb $outdir

# Copy zdock files 
cp zdock $outdir
cp uniCHARMM $outdir
#cp create_lig $outdir
#cp create.pl $outdir

cd $outdir 

# Now run the target against each decoy (which will include the real binder).
while read decoy_ppi_pair
do
	DECOY_PDBID=$(echo $decoy_ppi_pair | cut -d"_" -f1) 
	DECOY_CHAIN=$(echo $decoy_ppi_pair | cut -d"_" -f3) 
	decoy_filename=../../01-zdock_marked/$DECOY_PDBID\_$DECOY_CHAIN\_m.pdb
		
	runname=$PDBID\_$CHAIN1\_$DECOY_PDBID\_$DECOY_CHAIN
	# Run ZDock, only count the time of this step.
	/usr/bin/time -f 'user %U' -o $runname\_cpu_seconds.txt ./zdock -S 100 -o zdock_$runname\.out -R $PDBID\_$CHAIN1\_m\_bl.pdb -L $decoy_filename

done < ../../../benchmark_list.txt


# Remove zdock files
rm zdock
rm uniCHARMM
rm create_lig
rm create.pl 

