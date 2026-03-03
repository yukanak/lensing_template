import subprocess, os, sys
import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from astropy import constants as const
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

l = np.arange(0,4096+1)
ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
slpp = np.insert(slpp, 0, 0)
slpp = np.insert(slpp, 0, 0)
clkk = slpp[:4097] * (l*(l+1))**2/4
lmax = 4096
bins = np.logspace(np.log10(30), np.log10(4000), 51)
digitized = np.digitize(np.arange(6144), bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
yaml_file = 'bt_gmv3500.yaml'
yaml_file_standard_gaussianlcmbonly = 'bt_gmv3500_gaussianlcmbonly.yaml'
btmp = bt.btemplate(yaml_file)
btmp_standard_gaussianlcmbonly = bt.btemplate(yaml_file_standard_gaussianlcmbonly)
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size
nside = btmp.nside

N = 499
auto = 0
cross = 0
auto_in = 0
auto_reprocessed = 0
cross_reprocessed = 0
auto_in_reprocessed = 0
#for i in np.arange(1)+1:
for i in np.arange(N)+1:
    #a_standard, c_standard, a_in_standard = btmp.get_masked_spec(i, recompute=True, savefile=False)
    print(i)
    fname = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500/btmpl_specs_{i:04d}.npz'
    tmp = np.load(fname)
    auto += tmp['auto']
    cross += tmp['cross']
    auto_in += tmp['auto_in']
    fname = f'/scratch/users/yukanaka/temp/btemps/btmpl_specs_{i:04d}.npz'
    tmp = np.load(fname)
    auto_reprocessed += tmp['auto']
    cross_reprocessed += tmp['cross']
    auto_in_reprocessed += tmp['auto_in']
#a_oldplm, c_oldplm, a_in_oldplm = btmp_OLD.get_masked_spec(1)
auto /= N
cross /= N
auto_in /= N
auto_reprocessed /= N
cross_reprocessed /= N
auto_in_reprocessed /= N

# BIN
auto = np.array([auto[digitized == i].mean() for i in range(1, len(bins))])
cross = np.array([cross[digitized == i].mean() for i in range(1, len(bins))])
auto_in = np.array([auto_in[digitized == i].mean() for i in range(1, len(bins))])
auto_reprocessed = np.array([auto_reprocessed[digitized == i].mean() for i in range(1, len(bins))])
cross_reprocessed = np.array([cross_reprocessed[digitized == i].mean() for i in range(1, len(bins))])
auto_in_reprocessed = np.array([auto_in_reprocessed[digitized == i].mean() for i in range(1, len(bins))])

# TO INVESTIGATE SAMPLE VARIANCE, take sets of 10 of the Gaussian set
Nsims, Nbins = 499,len(bins)-1
group_size = 10
Ngroups = 49 #Nsims // group_size
Nuse = Ngroups * group_size
# Gaussian sims now for comparison
idxs = np.arange(499)+1
auto_standard = np.zeros((len(idxs),len(bins)-1),dtype=np.complex_)
cross_standard = np.zeros((len(idxs),len(bins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp.get_masked_spec(idx)
    auto_standard[ii,:] = [a_standard[digitized == i].mean() for i in range(1, len(bins))]
    cross_standard[ii,:] = [c_standard[digitized == i].mean() for i in range(1, len(bins))]
auto_mean_standard_gaussian = np.mean(auto_standard, axis=0)
auto_var_standard_gaussian = np.var(auto_standard, axis=0)
cross_mean_standard_gaussian = np.mean(cross_standard, axis=0)
cross_var_standard_gaussian = np.var(cross_standard, axis=0)
# randomize so groups aren't "seed 1-10, 11-20, ..."
rng = np.random.default_rng(0)
perm = rng.permutation(Nsims)[:Nuse]
auto_use = auto_standard[perm] # (Nuse, Nbins)
cross_use = cross_standard[perm] # (Nuse, Nbins)
# reshape into groups and take group means
auto_groups = auto_use.reshape(Ngroups, group_size, Nbins)
auto_mean10 = auto_groups.mean(axis=1) # (Ngroups, Nbins)
cross_groups = cross_use.reshape(Ngroups, group_size, Nbins)
cross_mean10 = cross_groups.mean(axis=1) # (Ngroups, Nbins)
# scatter of 10-sim means (this is the "error bar on a 10-sim mean")
auto_sigma10 = auto_mean10.std(axis=0)
cross_sigma10 = cross_mean10.std(axis=0)
#=============================================================================#

# plot
plt.figure(0)
plt.clf()
#for i in np.arange(499)+1:
#    fname = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500/btmpl_specs_{i:04d}.npz'
#    tmp = np.load(fname)['auto']
#    plt.plot(bin_centers, np.array([tmp[digitized == i].mean() for i in range(1, len(bins))]), color='lightgray', alpha=0.3, linestyle='-')
#for i in range(49):
#    if i == 0:
#        #plt.errorbar(bin_centers, cross_mean10[i,:], yerr=cross_sigma10, color='lightgray', alpha=0.3, linestyle='-',label='btemplate x input B cross, standard, 10 Gaussian sims')
#        plt.plot(bin_centers, cross_mean10[i,:], color='lightgray', alpha=0.3, linestyle='-',label='btemplate x input B cross, standard, 10 Gaussian sims')
#    else:
#        #plt.errorbar(bin_centers, cross_mean10[i,:], yerr=cross_sigma10, color='lightgray', alpha=0.3, linestyle='-')
#        plt.plot(bin_centers, cross_mean10[i,:], color='lightgray', alpha=0.3, linestyle='-')
# error band = expected scatter of a 10-sim mean
plt.fill_between(bin_centers,cross_mean_standard_gaussian-cross_sigma10,cross_mean_standard_gaussian+cross_sigma10,color='firebrick',alpha=0.3,label=r'$\pm 1\sigma$ (10-sim mean)')

plt.plot(bin_centers, auto[:lmax+1], color='firebrick', linestyle='-', label=f'btemplate auto, avg sims 1-499')
plt.plot(bin_centers, cross[:lmax+1], color='darkblue', linestyle='-', label=f'btemplate x input B cross, avg sims 1-499')

plt.plot(bin_centers, auto_in[:lmax+1], color='forestgreen', linestyle='-', label=f'input B auto, avg sims 1-499')
plt.plot(bin_centers, auto_in_reprocessed[:lmax+1], color='olive', linestyle='--', label=f'input B auto, avg sims 1-499, reprocessed')

plt.plot(bin_centers, auto_reprocessed[:lmax+1], color='salmon', linestyle='--', label='btemplate auto, avg sims 1-499, reprocessed')
plt.plot(bin_centers, cross_reprocessed[:lmax+1], color='cornflowerblue', linestyle='--', label='btemplate x input B cross, avg sims 1-499, reprocessed')

plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(3e-7,1.2e-6)
plt.legend(loc='lower left', fontsize='small')
plt.title(f'btemplate check')
plt.ylabel("$C_\ell^{BB}$")
plt.xlabel('$\ell$')
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(bin_centers, (auto_reprocessed/auto), color='firebrick', linestyle='--', alpha=0.8, label=f'btemplate auto, avg sims 1-499 reprocessed / orig')
plt.plot(bin_centers, (cross_reprocessed/cross), color='darkblue', linestyle='--', alpha=0.8, label=f'btemplate x input B cross, avg sims 1-499 reprocessed / orig')
plt.plot(bin_centers, (auto_in_reprocessed/auto_in), color='forestgreen', linestyle='--', alpha=0.8, label=f'input B auto, avg sims 1-499 reprocessed / orig')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0.95,1.3)
plt.legend(loc='lower left', fontsize='small')
plt.title(f'btemplate check')
plt.xlabel('$\ell$')
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')


