# Pablo Gainza 2019 - Assign the raw data from Intpred's webserver to each PDB's b factor column. 
# Also, verify the data. 
import numpy 
from Bio.PDB import * 
from IPython.core.debugger import set_trace


def parse_psiver_raw(pdbid, filename):
    pvfile = open(filename)
    lines = pvfile.readlines()
    line_ix = 0
    # Fourt line to verify pdbid - to make sure I made no mistakes.
    verify = lines[3].split()[2]
    verify_pdb = (''.join(verify[0:4])).upper()
    verify_chain = ''.join(verify[5])
    verify = '{}_{}'.format(verify_pdb,verify_chain)
    assert(verify == pdbid)
    print(verify)
    all_scores = {}

    # Ignore lines until we reach 'Residue Score'
    while not lines[line_ix].startswith('PRED'):
        line_ix += 1
    seq = []
    scores = []
    res_id = []
    for line in lines[line_ix:]:
        if line.startswith('PRED'):
            fields = line.split()
            aa = fields[3]
            score = float(fields[4])
            scores.append(score)
            seq.append(aa)
            res_id.append(int(fields[1]))

    return (seq, scores, res_id)

parser = PDBParser()
for line in open('psiver_list.txt').readlines():
    pdbid = line.rstrip()
    # Open the psiver prediction
    pred_fn = 'raw/'+pdbid
    seq, psiver_scores, res_id = parse_psiver_raw(pdbid, pred_fn)

    # Open the pdb file. 
    mystruct = parser.get_structure(pdbid, 'orig_pdbs/'+pdbid+'.pdb')
    resix = 0
    for model in mystruct:
        for chain in model: 
            for residue in chain: 
                aa_type_pdb = residue.get_resname()
                aa_type_pdb = Polypeptide.three_to_one(aa_type_pdb)
                # Check that the aa corresponds.
                assert(aa_type_pdb == seq[resix])
                for atom in residue: 
                    score = psiver_scores [resix]
                    atom.set_bfactor(score*100)
                resix += 1
    outio = PDBIO()
    outio.set_structure(mystruct)
    outio.save('psiver_pdbs/'+pdbid+'.pdb')
    print(line)
