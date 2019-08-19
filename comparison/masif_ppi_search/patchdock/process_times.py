lines = open("total_times.txt", "r").readlines()
total = 0.0
for line in lines:
    val = float(line.split(" ")[1])
    total += val
    print(val)
print('Total time (cpu minutes): {}'.format(total / 60))
