mylines = open('bc-100.out').readlines()

for line in mylines:
    # Go through every PDB in the line
    pdbid_chain = line.split()
    for pc in pdbid_chain:
        chain = pc.split('_')[1]
        if len(chain) == 1:
            print (pc)
            break


