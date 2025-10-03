import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/users/yukanaka/lensing_template/')
import btemplate as bt

yaml_file = '/home/users/yukanaka/lensing_template/bt_gmv3500.yaml'
yaml_file_corr = '/home/users/yukanaka/lensing_template/corr_vs_uncorr_noise_in_elm/bt_gmv3500_corr.yaml'
idxs = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_corr/signflip_midellcorr_20250520_goodidx.txt")
idxs = idxs[1:196].astype(int)
btmp_uncorr = bt.btemplate(yaml_file)
btmp_corr = bt.btemplate(yaml_file_corr)
lmax = 4096
l = np.arange(lmax+1)
lbins = np.logspace(np.log10(30),np.log10(1000),20)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(np.arange(6144), lbins)

auto_uncorr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_uncorr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_uncorr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_corr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_corr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_corr = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_uncorr, c_uncorr, a_in_uncorr = btmp_uncorr.get_masked_spec(idx)
    a_corr, c_corr, a_in_corr = btmp_corr.get_masked_spec(idx)
    auto_uncorr[ii,:] = [a_uncorr[digitized == i].mean() for i in range(1, len(lbins))]
    cross_uncorr[ii,:] = [c_uncorr[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_uncorr[ii,:] = [a_in_uncorr[digitized == i].mean() for i in range(1, len(lbins))]
    auto_corr[ii,:] = [a_corr[digitized == i].mean() for i in range(1, len(lbins))]
    cross_corr[ii,:] = [c_corr[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_corr[ii,:] = [a_in_corr[digitized == i].mean() for i in range(1, len(lbins))]
auto_mean_uncorr = np.mean(auto_uncorr, axis=0)
auto_var_uncorr = np.var(auto_uncorr, axis=0)
cross_mean_uncorr = np.mean(cross_uncorr, axis=0)
cross_var_uncorr = np.var(cross_uncorr, axis=0)
auto_input_mean_uncorr = np.mean(auto_in_uncorr, axis=0)
auto_input_var_uncorr = np.var(auto_in_uncorr, axis=0)
auto_mean_corr = np.mean(auto_corr, axis=0)
auto_var_corr = np.var(auto_corr, axis=0)
cross_mean_corr = np.mean(cross_corr, axis=0)
cross_var_corr = np.var(cross_corr, axis=0)
auto_input_mean_corr = np.mean(auto_in_corr, axis=0)
auto_input_var_corr = np.var(auto_in_corr, axis=0)

# Plot
plt.figure(0)
plt.clf()
plt.errorbar(bin_centers, auto_mean_uncorr, yerr=np.sqrt(auto_var_uncorr), color='firebrick', linestyle='-', label="lensing template auto, uncorr")
plt.errorbar(bin_centers, cross_mean_uncorr, yerr=np.sqrt(cross_var_uncorr), color='darkblue', linestyle='-', label="cross, uncorr")
plt.errorbar(bin_centers, auto_input_mean_uncorr, yerr=np.sqrt(auto_input_var_uncorr), color='forestgreen', linestyle='-', label="input B auto")
plt.errorbar(bin_centers, auto_mean_corr, yerr=np.sqrt(auto_var_corr), color='salmon', linestyle='--', label="lensing template auto, corr")
plt.errorbar(bin_centers, cross_mean_corr, yerr=np.sqrt(cross_var_corr), color='cornflowerblue', linestyle='--', label="cross, corr")
#plt.errorbar(bin_centers, auto_input_mean_corr, yerr=np.sqrt(auto_input_var_corr), color='lightgreen', linestyle='--', alpha=0.8, label="input B auto, corr")
plt.legend(loc='lower left', fontsize='x-small')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell$ [$\mu K^2$]")
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.show()
plt.savefig('/home/users/yukanaka/lensing_template/figs/btemplates_check.png',bbox_inches='tight')

plt.clf()
plt.plot(bin_centers, auto_var_uncorr, color='firebrick', linestyle='-', label="lensing template auto, uncorr")
plt.plot(bin_centers, cross_var_uncorr, color='darkblue', linestyle='-', label="cross, uncorr")
plt.plot(bin_centers, auto_input_var_uncorr, color='forestgreen', linestyle='-', label="input B auto")
plt.plot(bin_centers, auto_var_corr, color='salmon', linestyle='--', label="lensing template auto, corr")
plt.plot(bin_centers, cross_var_corr, color='cornflowerblue', linestyle='--', label="cross, corr")
plt.legend(loc='upper right', fontsize='x-small')
plt.xlabel(r"$\ell$")
plt.ylabel(r"Variance")
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.show()
plt.savefig('/home/users/yukanaka/lensing_template/figs/btemplates_var.png',bbox_inches='tight')

