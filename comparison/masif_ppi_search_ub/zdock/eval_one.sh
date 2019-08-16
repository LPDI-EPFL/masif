PDBID_CHAIN1_CHAIN2=$1
PDBID=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f1)
CHAIN1=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f2)
CHAIN2=$(echo $PDBID_CHAIN1_CHAIN2 | cut -d"_" -f3)

outdir=03-results/$PDBID_CHAIN1_CHAIN2/

cp uniCHARMM $outdir
cp create_lig $outdir
cp create.pl $outdir

# Copy evaluation script
cp eval_zdock.py $outdir
cd $outdir

# Run create to create the pdbs for the wildtype.
./create.pl zdock_$PDBID\_$CHAIN1\_$PDBID\_$CHAIN2\.out 1000

# Evaluate the prediction in the context of all predictions.
python3 eval_zdock.py $PDBID_CHAIN1_CHAIN2 $PDBID\_$CHAIN1\_m\_bl.pdb $PDBID\_$CHAIN2\_m.pdb | tee results.txt

rm complex*.pdb
