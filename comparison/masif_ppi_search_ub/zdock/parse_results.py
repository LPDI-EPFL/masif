import os
import numpy as np
stats = []
cpu_seconds = []

for dirname in os.listdir('03-results/'):
    full_dirname = os.path.join('03-results/',dirname)
    if not os.path.exists(full_dirname+'/results.txt'):
        continue
    results = open(full_dirname+'/results.txt').readlines()
    for time_fn in os.listdir(full_dirname):
        if 'cpu_seconds' in time_fn: 
            cpu_seconds_file = open(os.path.join(full_dirname, time_fn),'r')
            for line in cpu_seconds_file: 
                if line.startswith('user'):
                    secs = float(line.split()[1])
                    cpu_seconds.append(secs)
                    break
    try:
        overall_rank = results[1].rstrip().split()[2]
        print('{} {}'.format(dirname, overall_rank))
    except: 
        print('Error in {}'.format(dirname))
        continue
    if 'N/D' in overall_rank:
        complex_val = float('inf')
    else:
        complex_val = float(overall_rank)
    stats.append(complex_val)

stats.sort()
stats = np.array(stats)
print(stats)
print('Total: {}'.format(len(stats)))
print('Ranked within top 1000: {}'.format(np.sum(stats<1000)))
print('Ranked within top 100: {}'.format(np.sum(stats<100)))
print('Ranked within top 10: {}'.format(np.sum(stats<10)))
print('Ranked #1: {}'.format(np.sum(stats<2)))
print('Not ranked at all: {}'.format(np.sum(stats>100000000000)))
print('Total time: {} minutes'.format(np.sum(cpu_seconds)/60)) 
