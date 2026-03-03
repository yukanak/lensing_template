import numpy as np
import matplotlib.pyplot as plt

weightsdir = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/gnilc_pr3_cib_pr4_kappa_standard/'
lmax = 2000
l = np.arange(lmax+1)
bin_edges = np.array([0, 50, 100, 500, 1000, 1500, 2001])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Check scatter of 10 runs (seeded differently) on the same sim (i.e., optimizer stochasticity)
seeds = np.arange(10)+1
A1 = np.zeros((len(seeds),6))
A2 = np.zeros((len(seeds),6))
for i,seed in enumerate(seeds):
    A1[i,:] = np.load(weightsdir+f'/A1_bins_klm1_weight_tuned_sim1_seed{seed}.npy')
    A2[i,:] = np.load(weightsdir+f'/A2_bins_klm2_weight_tuned_sim1_seed{seed}.npy')
A1_mean_10seeds = np.mean(A1, axis=0)
A1_std_10seeds = np.std(A1, axis=0)
A2_mean_10seeds = np.mean(A2, axis=0)
A2_std_10seeds = np.std(A2, axis=0)

# Check scatter of results (seeded randomly) on 10 different sims (i.e., realization scatter)
# "Do different sims prefer different weights?"
idxs = np.arange(10)+1
A1 = np.zeros((len(idxs),6))
A2 = np.zeros((len(idxs),6))
for i,idx in enumerate(idxs):
    A1[i,:] = np.load(weightsdir+f'/A1_bins_klm1_weight_tuned_sim{idx}.npy')
    A2[i,:] = np.load(weightsdir+f'/A2_bins_klm2_weight_tuned_sim{idx}.npy')
A1_mean_10sims = np.mean(A1, axis=0)
A1_std_10sims = np.std(A1, axis=0)
A2_mean_10sims = np.mean(A2, axis=0)
A2_std_10sims = np.std(A2, axis=0)

# Plot
factors = [0.94,0.98,1.02,1.06]
plt.figure(0)
plt.clf()
plt.errorbar(bin_centers*factors[0], A1_mean_10sims, yerr=A1_std_10sims, color='firebrick', linestyle='-', label="A1, 10 sims")
plt.errorbar(bin_centers*factors[1], A2_mean_10sims, yerr=A2_std_10sims, color='darkblue', linestyle='-', label="A2, 10 sims")
plt.errorbar(bin_centers*factors[2], A1_mean_10seeds, yerr=A1_std_10seeds, color='salmon', linestyle='-', label="A1, 10 seeds")
plt.errorbar(bin_centers*factors[3], A2_mean_10seeds, yerr=A2_std_10seeds, color='cornflowerblue', linestyle='-', label="A2, 10 seeds")
plt.legend(loc='lower left', fontsize='small')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$A$")
plt.xscale('log')
#plt.yscale('log')
plt.xlim(10,2000)
plt.show()
plt.savefig('figs/temp.png',bbox_inches='tight')

