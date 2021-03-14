import numpy as np
from IPython.core.debugger import set_trace

theta_old = np.load('theta_old.npy')
rho_old = np.load('rho_old.npy')
neigh_indices_old = np.load('neigh_indices_old.npy', allow_pickle=True)

theta_new = np.load('theta_new.npy')
rho_new = np.load('rho_new.npy')
neigh_indices_new = np.load('neigh_indices_new.npy')

all_old = []
all_new = []
mean_square_error = []
for i in range(len(neigh_indices_old)):
    # Randomly rotate each patch 
    n = len(neigh_indices_old[i])
    rand_rot = np.random.random()*2*np.pi
    theta_new[i,:n] = np.fmod(theta_new[i, :n] + rand_rot, 2*np.pi)
    theta_new[i, 0] = 0
    for j in range(len(neigh_indices_old[i])):
        assert(neigh_indices_old[i][j] == neigh_indices_new[i][j])
        try:
            assert(np.abs(rho_new[i][j] - rho_old[i][j])< 1e-7)
        except:
            print(rho_new[i][j], rho_old[i][j])
        all_old.append(theta_old[i][j])
        all_new.append(theta_new[i][j])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.hist(all_old, bins=36)
plt.hist(all_new, bins=36, alpha=0.75)
plt.savefig('theta_angle_distribution.png')
plt.close()


fixed_error = []
non_fixed_error = []
for i in range(len(neigh_indices_old)):
    angle_diff = []
    for j in range(1,len(neigh_indices_old[i])):
        x = theta_old[i][j]
        y = theta_new[i][j]
        angle_diff.append(np.arctan2(np.sin(x-y), np.cos(x-y)))
# Now correct the patch. 
    theta_new[i] = np.fmod(theta_new[i]+np.median(angle_diff) + 2*np.pi , 2*np.pi)

    fixed_diff = []
    for j in range(len(neigh_indices_old[i])):
        x = theta_old[i][j]
        y = theta_new[i][j]
        fixed_diff.append(np.arctan2(np.sin(x-y), np.cos(x-y)))

    fixed_error.append(np.sqrt(np.mean(np.square(fixed_diff))))
    non_fixed_error.append(np.sqrt(np.mean(np.square(angle_diff))))

print('Median non-fixed error: {}'.format(np.median(non_fixed_error)))
print('Median fixed error: {}'.format(np.median(fixed_error)))

plt.hist(non_fixed_error, bins=36)
plt.hist(fixed_error, bins=36, alpha=0.75)
plt.savefig('error_in_theta.png')

