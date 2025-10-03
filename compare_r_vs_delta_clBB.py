import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
import camb

lmax = 4096
l = np.arange(lmax+1)
lbins = np.logspace(np.log10(30),np.log10(1000),20)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(np.arange(6144), lbins)
yaml_file = 'bt_gmv3500.yaml'
yaml_file_prfhrd = 'bt_gmv3500_prfhrd.yaml'
yaml_file_pp = 'bt_gmv3500_pp.yaml'
btmp_standard = bt.btemplate(yaml_file)
btmp_prfhrd = bt.btemplate(yaml_file_prfhrd)
btmp_pp = bt.btemplate(yaml_file_pp)

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=1) # Primordial spectrum with r = 1
pars.WantTensors = True # Get tensor Cls
pars.DoLensing = False # Turn OFF lensing
pars.set_for_lmax(6143, lens_potential_accuracy=0)
results = camb.get_results(pars)
cls = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
ell = np.arange(cls['tensor'].shape[0]) # Tensor-only spectra (unlensed)
ClTT_tens, ClEE_tens, ClBB_tens, ClTE_tens = cls['tensor'].T
binned_clbb_tens = np.array([ClBB_tens[digitized == i].mean() for i in range(1, len(lbins))])

# Get delta(clbb) between lensing templates from different tracers
a_standard, c_standard, _ = btmp_standard.get_masked_spec(0)
auto_standard = np.array([a_standard[digitized == i].mean() for i in range(1, len(lbins))])
#cross_standard = np.array([c_standard[digitized == i].mean() for i in range(1, len(lbins))])
a_prfhrd, _, _ = btmp_prfhrd.get_masked_spec(0)
auto_prfhrd = np.array([a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))])
a_pp, _, _ = btmp_pp.get_masked_spec(0)
auto_pp = np.array([a_pp[digitized == i].mean() for i in range(1, len(lbins))])
idxs = np.arange(499)+1
diff_auto_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_sim_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_sim_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]

    auto_sim_standard[ii,:] = a_standard
    cross_sim_standard[ii,:] = [c_standard[digitized == i].mean() for i in range(1, len(lbins))]
    diff_auto_prfhrd[ii,:] = np.array(a_standard) - np.array(a_prfhrd)
    diff_auto_pp[ii,:] = np.array(a_standard) - np.array(a_pp)
    diff_auto_pp_prfhrd[ii,:] = np.array(a_prfhrd) - np.array(a_pp)
auto_sim_mean_standard = np.mean(auto_sim_standard, axis=0)
cross_sim_mean_standard = np.mean(cross_sim_standard, axis=0)
auto_std_prfhrd = np.std(diff_auto_prfhrd, axis=0)
auto_std_pp = np.std(diff_auto_pp, axis=0)
auto_std_pp_prfhrd = np.std(diff_auto_pp_prfhrd, axis=0)
auto_mean_prfhrd = np.mean(diff_auto_prfhrd, axis=0) # sim mean to subtract
auto_mean_pp = np.mean(diff_auto_pp, axis=0) # sim mean to subtract
auto_mean_pp_prfhrd = np.mean(diff_auto_pp_prfhrd, axis=0) # sim mean to subtract
diff_prfhrd = ((np.array(auto_standard) - np.array(auto_prfhrd)) - auto_mean_prfhrd)
diff_pp = ((np.array(auto_standard) - np.array(auto_pp)) - auto_mean_pp)

# Plot equivalent r assuming primordial B spectrum scales linearly with r: clbb = r * clbb(r=1)
r_eq_prfhrd = diff_prfhrd / binned_clbb_tens
r_eq_pp = diff_pp / binned_clbb_tens

# The difference between standard and hardened tracers corresponds to a false signal at the level of what r at which ell values?
plt.figure(0)
plt.clf()
plt.plot(bin_centers, r_eq_prfhrd, color='firebrick', linestyle='-', label="prfhrd")
plt.plot(bin_centers, r_eq_pp, color='darkblue', linestyle='-', label="pol-only")
plt.axhline(0.01, color='gray', ls='--', label='r = 0.01')
plt.axhline(0.001, color='gray', ls=':', label='r = 0.001')
plt.legend(loc='upper right', fontsize='medium')
plt.xscale('log')
plt.xlabel(r"$\ell$")
plt.ylabel(r"equivalent $r$ as $\Delta C_\ell^{BB} / C_\ell^{BB}(r = 1)$")
plt.xlim(30,400)
plt.ylim(-0.05, 0.015)
plt.savefig('figs/equivalent_tensor_to_scalar_ratio.png',bbox_inches='tight')

# https://sptlocal.grid.uchicago.edu/~wlwu/20221117_btpl_ngfgbias/
#cross_bias = -2*(cross_standard - cross_sim_mean_standard)
auto_bias = auto_standard - auto_sim_mean_standard
#total_bias = cross_bias + auto_bias
plt.clf()
plt.axhline(0, color='gray', ls='--',)
#plt.plot(bin_centers, cross_bias, color='blue', linestyle='-', label=r"GMV standard $\phi$: $\Delta (\hat{B} \times B^{lens})$")
plt.plot(bin_centers, auto_bias, color='orange', linestyle='-', label=r"GMV standard $\phi$: $\Delta (\hat{B} \times \hat{B})$")
#plt.plot(bin_centers, total_bias, color='green', linestyle='-', label=r"GMV standard $\phi$: $\Delta C_{\ell}^{BB,del}$")
plt.legend(loc='upper right', fontsize='medium')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\Delta C_\ell^{BB} [\mu K^2]$")
plt.xlim(0,1000)
plt.ylim(-0.6e-7, 1.2e-7)
plt.savefig('figs/clbb_fg_bias.png',bbox_inches='tight')

'''
plt.clf()
x = np.linspace(0, 1.0e-7, 200)
pick = [2, 5, 9, 12]  # change as you like
for b in pick:
    r_eq = x / binned_clbb_tens[b]
    plt.plot(x, r_eq, label=fr'bin {b}: $\ell \!\in$ [{lbins[b]:.0f},{lbins[b+1]:.0f})')
plt.xlabel(r'$\Delta C^{BB}$  [$\mu{\rm K}^2$]')
plt.ylabel(r'equivalent $r$  ($\Delta C^{BB}/C^{BB}(r{=}1)$)')
plt.title(r'mapping between $\Delta C^{BB}$ and $r$ (per $\ell$-bin)')
plt.grid(alpha=0.3)
plt.legend(fontsize='small')
plt.savefig('figs/equivalent_tensor_to_scalar_ratio_theory.png',bbox_inches='tight')
'''

