# Pablo Gainza 2019 - Assign the raw data from Intpred's webserver to each PDB's b factor column. 
# Also, verify the data. 
import numpy 
from Bio.PDB import * 
from IPython.core.debugger import set_trace


def parse_intpred_raw(pdbid, filename):
    ipfile = open(filename)
    lines = ipfile.readlines()
    line_ix = 0
    # First line to verify pdbid - to make sure I made no mistakes.
    verify = lines[0]
    verify = lines[0].split()
    verify_pdb = (''.join(verify[5][0:4])).upper()
    verify_chain = ''.join(verify[5][4])
    verify = '{}_{}'.format(verify_pdb,verify_chain)
    assert(verify == pdbid)
    print(verify)
    all_scores = {}

    # Ignore lines until we reach 'Residue Score'
    while "Residue Score" not in lines[line_ix]:
        line_ix += 1
    line_ix += 1
    for line in lines[line_ix:]:
        fields = line.split()
        if len(fields) > 0:
            chain_id = fields[0][0]
            residue = int(fields[0][1:])
            score = float(fields[1])
            all_scores[(chain_id, residue)] = score

    return all_scores

parser = PDBParser()
for line in open('intpred_list.txt').readlines():
    pdbid = line.rstrip()
    # Open the intpred prediction
    pred_fn = 'raw/'+pdbid
    try:
        intpred_scores = parse_intpred_raw(pdbid, pred_fn)
    except:
        continue

    # Open the pdb file. 
    mystruct = parser.get_structure(pdbid, 'orig_pdbs/'+pdbid+'.pdb')
    for model in mystruct:
        for chain in model: 
            for residue in chain: 
                for atom in residue: 
                    key = (chain.get_id(), int(residue.get_id()[1]))
                    if key in intpred_scores:
                        atom.set_bfactor(intpred_scores[key]*100)
                        #atom.set_bfactor(99)
                    else:
                        atom.set_bfactor(-99)
    outio = PDBIO()
    outio.set_structure(mystruct)
    outio.save('intpred_pdbs/'+pdbid+'.pdb')
    print(line)
