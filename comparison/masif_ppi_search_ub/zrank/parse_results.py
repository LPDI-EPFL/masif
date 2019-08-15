import os 
import numpy as np
all_ranks = []
user_time = 0.0
for results_dir in os.listdir('results/'):
    full_dir = os.path.join('results/', results_dir)
    results_file = os.path.join(full_dir, 'results.txt')
    if os.path.exists(results_file):
        results_line = open(results_file).readlines()
        fields = results_line[0].split()

        if fields[3] == 'N/D': 
            rank = 20001
        else:
            rank = int(fields[3])
        all_ranks.append(rank)
    for time_file in os.listdir(full_dir):
        if 'cpu_seconds' in time_file:
            cpu_time_lines = open(os.path.join(full_dir, time_file)).readlines()
            for line in cpu_time_lines:
                if line.startswith('user'):
                    user_time+= float(line.split()[1])


all_ranks = np.array(all_ranks)

print("Num in top 10: {}".format(np.sum(all_ranks<10)))
print("Num in top 100: {}".format(np.sum(all_ranks<100)))
print("Num in top 1000: {}".format(np.sum(all_ranks<1000)))
print("ZRank time: {:.2f} minutes".format(user_time/60))
