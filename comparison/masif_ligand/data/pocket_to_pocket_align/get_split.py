"""
get_split.py: Once a pocket has been align, print the pockets that do not align at 0.5 or more 
                to at least one pocket in the training set. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

import os 
from IPython.core.debugger import set_trace
import numpy as np
for fn in os.listdir('out_pocket/'):
    outlines = open('out_pocket/'+fn, 'r').readlines()
    train_id = []
    test_id = []
    tmscore = []
    for line in outlines:
        fields = line.split(',')
        train_id.append(fields[0])
        test_id.append(fields[1])
        tmscore.append(float(fields[2]))
    tmscore = np.array(tmscore)
    tmscore = np.nan_to_num(tmscore) 
    try:
        max_ix = np.argmax(tmscore)
    except:
        set_trace()
    if tmscore[max_ix] < 0.50:
        print('{},{},{}'.format(test_id[max_ix], train_id[max_ix], tmscore[max_ix] ))
    #print('{},{},{} -- Failures: {} out of {}'.format(train_id[max_ix], test_id[max_ix], tmscore[max_ix], np.sum(tmscore==0), len(tmscore)))



