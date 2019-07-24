lines = open("total_times.txt", "r").readlines()
total_time = 0.0
for line in lines:
    time = line.split()[1]
    total_time += float(time)

print(total_time)
