import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
import matplotlib.pyplot as plt
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
from scipy.interpolate import interp1d

#yaml_file_agoraGfgs = 'bt_gmv3500_prfhrd_agoraGfgs.yaml'
#btmp_agoraGfgs = bt.btemplate(yaml_file_agoraGfgs)
yaml_file = 'bt_gmv3500_prfhrd.yaml'
btmp = bt.btemplate(yaml_file)
yaml_file_standard = 'bt_gmv3500.yaml'
btmp_standard = bt.btemplate(yaml_file_standard)
yaml_file_pp = 'bt_gmv3500_pp.yaml'
btmp_pp = bt.btemplate(yaml_file_pp)
yaml_file_standard_agoraGfgs = 'bt_gmv3500_agoraGfgs.yaml'
btmp_standard_agoraGfgs = bt.btemplate(yaml_file_standard_agoraGfgs)
yaml_file_standard_agoraGfgs_lmax1000 = 'bt_gmv3500_agoraGfgs_lmax1000.yaml'
btmp_standard_agoraGfgs_lmax1000 = bt.btemplate(yaml_file_standard_agoraGfgs_lmax1000)
nside = btmp.nside
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size
lbins = np.logspace(np.log10(30),np.log10(1000),20)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(np.arange(6144), lbins)

#=============================================================================#

idxs = np.arange(10) + 5001
auto_standard_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_standard_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_standard_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_standard_agoraGfgs_lmax1000 = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_standard_agoraGfgs_lmax1000 = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_standard_agoraGfgs_lmax1000 = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_prfhrd_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_prfhrd_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_prfhrd_agoraGfgs = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard_agoraGfgs, c_standard_agoraGfgs, a_in_standard_agoraGfgs = btmp_standard_agoraGfgs.get_masked_spec(idx)
    auto_standard_agoraGfgs[ii,:] = [a_standard_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    cross_standard_agoraGfgs[ii,:] = [c_standard_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_standard_agoraGfgs[ii,:] = [a_in_standard_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    #a_prfhrd_agoraGfgs, c_prfhrd_agoraGfgs, a_in_prfhrd_agoraGfgs = btmp_agoraGfgs.get_masked_spec(idx)
    #auto_prfhrd_agoraGfgs[ii,:] = [a_prfhrd_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    #cross_prfhrd_agoraGfgs[ii,:] = [c_prfhrd_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    #auto_in_prfhrd_agoraGfgs[ii,:] = [a_in_prfhrd_agoraGfgs[digitized == i].mean() for i in range(1, len(lbins))]
    a_standard_agoraGfgs_lmax1000, c_standard_agoraGfgs_lmax1000, a_in_standard_agoraGfgs_lmax1000 = btmp_standard_agoraGfgs_lmax1000.get_masked_spec(idx)
    auto_standard_agoraGfgs_lmax1000[ii,:] = [a_standard_agoraGfgs_lmax1000[digitized == i].mean() for i in range(1, len(lbins))]
    cross_standard_agoraGfgs_lmax1000[ii,:] = [c_standard_agoraGfgs_lmax1000[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_standard_agoraGfgs_lmax1000[ii,:] = [a_in_standard_agoraGfgs_lmax1000[digitized == i].mean() for i in range(1, len(lbins))]
auto_mean_standard_agoraGfgs = np.mean(auto_standard_agoraGfgs, axis=0)
auto_var_standard_agoraGfgs = np.var(auto_standard_agoraGfgs, axis=0)
cross_mean_standard_agoraGfgs = np.mean(cross_standard_agoraGfgs, axis=0)
cross_var_standard_agoraGfgs = np.var(cross_standard_agoraGfgs, axis=0)
auto_input_mean_standard_agoraGfgs = np.mean(auto_in_standard_agoraGfgs, axis=0)
auto_input_var_standard_agoraGfgs = np.var(auto_in_standard_agoraGfgs, axis=0)
auto_mean_prfhrd_agoraGfgs = np.mean(auto_prfhrd_agoraGfgs, axis=0)
auto_var_prfhrd_agoraGfgs = np.var(auto_prfhrd_agoraGfgs, axis=0)
cross_mean_prfhrd_agoraGfgs = np.mean(cross_prfhrd_agoraGfgs, axis=0)
cross_var_prfhrd_agoraGfgs = np.var(cross_prfhrd_agoraGfgs, axis=0)
auto_input_mean_prfhrd_agoraGfgs = np.mean(auto_in_prfhrd_agoraGfgs, axis=0)
auto_input_var_prfhrd_agoraGfgs = np.var(auto_in_prfhrd_agoraGfgs, axis=0)
auto_mean_standard_agoraGfgs_lmax1000 = np.mean(auto_standard_agoraGfgs_lmax1000, axis=0)
auto_var_standard_agoraGfgs_lmax1000 = np.var(auto_standard_agoraGfgs_lmax1000, axis=0)
cross_mean_standard_agoraGfgs_lmax1000 = np.mean(cross_standard_agoraGfgs_lmax1000, axis=0)
cross_var_standard_agoraGfgs_lmax1000 = np.var(cross_standard_agoraGfgs_lmax1000, axis=0)
auto_input_mean_standard_agoraGfgs_lmax1000 = np.mean(auto_in_standard_agoraGfgs_lmax1000, axis=0)
auto_input_var_standard_agoraGfgs_lmax1000 = np.var(auto_in_standard_agoraGfgs_lmax1000, axis=0)

#=============================================================================#

idxs = np.arange(10) + 5001
auto_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    auto_standard[ii,:] = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    cross_standard[ii,:] = [c_standard[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_standard[ii,:] = [a_in_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp.get_masked_spec(idx)
    auto_prfhrd[ii,:] = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    cross_prfhrd[ii,:] = [c_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_prfhrd[ii,:] = [a_in_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    auto_pp[ii,:] = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]
    cross_pp[ii,:] = [c_pp[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_pp[ii,:] = [a_in_pp[digitized == i].mean() for i in range(1, len(lbins))]
auto_mean_standard = np.mean(auto_standard, axis=0)
auto_var_standard = np.var(auto_standard, axis=0)
cross_mean_standard = np.mean(cross_standard, axis=0)
cross_var_standard = np.var(cross_standard, axis=0)
auto_input_mean_standard = np.mean(auto_in_standard, axis=0)
auto_input_var_standard = np.var(auto_in_standard, axis=0)
auto_mean_prfhrd = np.mean(auto_prfhrd, axis=0)
auto_var_prfhrd = np.var(auto_prfhrd, axis=0)
cross_mean_prfhrd = np.mean(cross_prfhrd, axis=0)
cross_var_prfhrd = np.var(cross_prfhrd, axis=0)
auto_input_mean_prfhrd = np.mean(auto_in_prfhrd, axis=0)
auto_input_var_prfhrd = np.var(auto_in_prfhrd, axis=0)
auto_mean_pp = np.mean(auto_pp, axis=0)
auto_var_pp = np.var(auto_pp, axis=0)
cross_mean_pp = np.mean(cross_pp, axis=0)
cross_var_pp = np.var(cross_pp, axis=0)
auto_input_mean_pp = np.mean(auto_in_pp, axis=0)
auto_input_var_pp = np.var(auto_in_pp, axis=0)

#=============================================================================#

# Gaussian sims now for comparison
idxs = np.arange(499)+1
auto_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
auto_in_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    auto_standard[ii,:] = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    cross_standard[ii,:] = [c_standard[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_standard[ii,:] = [a_in_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp.get_masked_spec(idx)
    auto_prfhrd[ii,:] = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    cross_prfhrd[ii,:] = [c_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_prfhrd[ii,:] = [a_in_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    auto_pp[ii,:] = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]
    cross_pp[ii,:] = [c_pp[digitized == i].mean() for i in range(1, len(lbins))]
    auto_in_pp[ii,:] = [a_in_pp[digitized == i].mean() for i in range(1, len(lbins))]
auto_mean_standard_gaussian = np.mean(auto_standard, axis=0)
auto_var_standard_gaussian = np.var(auto_standard, axis=0)
cross_mean_standard_gaussian = np.mean(cross_standard, axis=0)
cross_var_standard_gaussian = np.var(cross_standard, axis=0)
auto_input_mean_standard_gaussian = np.mean(auto_in_standard, axis=0)
auto_input_var_standard_gaussian = np.var(auto_in_standard, axis=0)
auto_mean_prfhrd_gaussian = np.mean(auto_prfhrd, axis=0)
auto_var_prfhrd_gaussian = np.var(auto_prfhrd, axis=0)
cross_mean_prfhrd_gaussian = np.mean(cross_prfhrd, axis=0)
cross_var_prfhrd_gaussian = np.var(cross_prfhrd, axis=0)
auto_input_mean_prfhrd_gaussian = np.mean(auto_in_prfhrd, axis=0)
auto_input_var_prfhrd_gaussian = np.var(auto_in_prfhrd, axis=0)
auto_mean_pp_gaussian = np.mean(auto_pp, axis=0)
auto_var_pp_gaussian = np.var(auto_pp, axis=0)
cross_mean_pp_gaussian = np.mean(cross_pp, axis=0)
cross_var_pp_gaussian = np.var(cross_pp, axis=0)
auto_input_mean_pp_gaussian = np.mean(auto_in_pp, axis=0)
auto_input_var_pp_gaussian = np.var(auto_in_pp, axis=0)

#=============================================================================#

# Get rho
rho_gaussian_std = cross_mean_standard_gaussian / np.sqrt(auto_mean_standard_gaussian * auto_input_mean_standard_gaussian) 
rho_gaussian_prfhrd = cross_mean_prfhrd_gaussian / np.sqrt(auto_mean_prfhrd_gaussian * auto_input_mean_prfhrd_gaussian) 
rho_gaussian_pp = cross_mean_pp_gaussian / np.sqrt(auto_mean_pp_gaussian * auto_input_mean_pp_gaussian) 
rho_agora_std = cross_mean_standard / np.sqrt(auto_mean_standard * auto_input_mean_standard) 
rho_agora_prfhrd = cross_mean_prfhrd / np.sqrt(auto_mean_prfhrd * auto_input_mean_prfhrd) 
rho_agora_pp = cross_mean_pp / np.sqrt(auto_mean_pp * auto_input_mean_pp) 
rho_agora_std_agoraGfgs  = cross_mean_standard_agoraGfgs  / np.sqrt(auto_mean_standard_agoraGfgs  * auto_input_mean_standard_agoraGfgs ) 
#rho_agora_prfhrd_agoraGfgs  = cross_mean_prfhrd_agoraGfgs  / np.sqrt(auto_mean_prfhrd_agoraGfgs  * auto_input_mean_prfhrd_agoraGfgs ) 

#=============================================================================#

ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',lmax=1000)

plt.figure(0)
plt.clf()
plt.plot(bin_centers, rho_gaussian_std, color='firebrick', alpha=1, label='rho Gaussian standard')
plt.plot(bin_centers, rho_gaussian_prfhrd, color='forestgreen', alpha=1, linestyle='-', label='rho Gaussian prfhrd')
plt.plot(bin_centers, rho_gaussian_pp, color='darkblue', alpha=1, linestyle='-', label='rho Gaussian pol-only')
plt.plot(bin_centers, rho_agora_std, color='salmon', alpha=1, linestyle='--', label='rho Agora standard')
plt.plot(bin_centers, rho_agora_prfhrd, color='lightgreen', alpha=1, linestyle='--', label='rho Agora prfhrd')
plt.plot(bin_centers, rho_agora_pp, color='cornflowerblue', alpha=1, linestyle='--', label='rho Agora pol-only')
plt.grid(True, linestyle="--", alpha=0.5)                                       
plt.legend(loc='upper right', fontsize='small')                                 
plt.xscale('log')                                                               
plt.xlim(10,1500)                                                               
plt.tight_layout()                                                              
plt.ylim(0.4,0.8)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\rho$")
plt.savefig('figs/btemplates_check_prfhrd_agora.png',bbox_inches='tight')

plt.figure(0)
plt.clf()
plt.plot(ell, slbb, color='gray', linestyle=':', label="CAMB theory slbb")
#plt.plot(bin_centers, clbb_fullsky, color='orange', linestyle='-', label="input B auto, full-sky Agora 5001 (unrotated, beam applied)")
#plt.plot(bin_centers, clbb_fullsky, color='orange', linestyle='-', label="input B auto, full-sky Agora 5001 (unrotated, beam NOT applied)")

#plt.errorbar(bin_centers, auto_mean_standard_gaussian, yerr=np.sqrt(auto_var_standard_gaussian), color='firebrick', linestyle='-', label="btemplate auto, standard, Gaussian sims")
#plt.errorbar(bin_centers, auto_mean_prfhrd_gaussian, yerr=np.sqrt(auto_var_prfhrd_gaussian), color='forestgreen', linestyle='-', label="btemplate auto, prfhrd, Gaussian sims")
#plt.errorbar(bin_centers, auto_mean_pp_gaussian, yerr=np.sqrt(auto_var_pp_gaussian), color='darkblue', linestyle='-', label="btemplate auto, pol-only, Gaussian sims")
plt.errorbar(bin_centers, cross_mean_standard_gaussian, yerr=np.sqrt(cross_var_standard_gaussian), color='firebrick', linestyle='-', label="btemplate x input B cross, standard, Gaussian sims")
plt.errorbar(bin_centers, cross_mean_prfhrd_gaussian, yerr=np.sqrt(cross_var_prfhrd_gaussian), color='forestgreen', linestyle='-', label="btemplate x input B cross, prfhrd, Gaussian sims")
plt.errorbar(bin_centers, cross_mean_pp_gaussian, yerr=np.sqrt(cross_var_pp_gaussian), color='darkblue', linestyle='-', label="btemplate x input B cross, pol-only, Gaussian sims")

#plt.errorbar(bin_centers, auto_input_mean_standard_gaussian, yerr=np.sqrt(auto_input_var_standard_gaussian), color='olive', linestyle='-', label="input B auto, Gaussian sims")
#plt.errorbar(bin_centers, auto_input_mean_standard, yerr=np.sqrt(auto_input_var_standard), color='forestgreen', linestyle='-', label="input B auto, Agora sims")

#plt.errorbar(bin_centers, auto_mean_standard, yerr=np.sqrt(auto_var_standard), color='salmon', linestyle='--', label="btemplate auto, standard, Agora sims")
#plt.errorbar(bin_centers, auto_mean_prfhrd, yerr=np.sqrt(auto_var_prfhrd), color='lightgreen', linestyle='--', label="btemplate auto, prfhrd, Agora sims")
#plt.errorbar(bin_centers, auto_mean_pp, yerr=np.sqrt(auto_var_pp), color='cornflowerblue', linestyle='--', label="btemplate auto, pol-only, Agora sims")
#plt.errorbar(bin_centers, auto_mean_standard_agoraGfgs, yerr=np.sqrt(auto_var_standard_agoraGfgs), color='mediumpurple', linestyle='--', label="btemplate x input B cross, standard, Agora WITH G FG sims")
#plt.errorbar(bin_centers, auto_mean_standard_agoraGfgs_lmax1000, yerr=np.sqrt(auto_var_standard_agoraGfgs_lmax1000), color='plum', linestyle='--', label="btemplate x input B cross, standard, Agora WITH G FG sims, lmax=1000")
plt.errorbar(bin_centers, cross_mean_prfhrd_agoraGfgs , yerr=np.sqrt(cross_var_prfhrd_agoraGfgs), color='mediumseagreen', linestyle='--', label="btemplate x input B cross, prfhrd, Agora WITH G FG sims")
plt.errorbar(bin_centers, cross_mean_standard, yerr=np.sqrt(cross_var_standard), color='salmon', linestyle='--', label="btemplate x input B cross, standard, Agora sims")
plt.errorbar(bin_centers, cross_mean_prfhrd, yerr=np.sqrt(cross_var_prfhrd), color='lightgreen', linestyle='--', label="btemplate x input B cross, prfhrd, Agora sims")
plt.errorbar(bin_centers, cross_mean_pp, yerr=np.sqrt(cross_var_pp), color='cornflowerblue', linestyle='--', label="btemplate x input B cross, pol-only, Agora sims")
plt.errorbar(bin_centers, cross_mean_standard_agoraGfgs, yerr=np.sqrt(cross_var_standard_agoraGfgs), color='mediumpurple', linestyle='--', label="btemplate x input B cross, standard, Agora WITH G FG sims")
plt.errorbar(bin_centers, cross_mean_standard_agoraGfgs_lmax1000, yerr=np.sqrt(cross_var_standard_agoraGfgs_lmax1000), color='plum', linestyle='--', label="btemplate x input B cross, standard, Agora WITH G FG sims, lmax=1000")
#plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
plt.legend(loc='lower left', fontsize='x-small')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
#plt.ylim(1e-7,3e-6)
plt.ylim(3e-7,1.2e-6)
plt.show()
plt.savefig('figs/btemplates_check_prfhrd_agora.png',bbox_inches='tight')

#=============================================================================#
'''
def moving_average_logx(x, y, window_width_log):
    """
    Smooth y(x) with a moving average uniform in log(x).
    window_width_log is the half-width in log10(x).
    """
    lx = np.log10(x)
    smooth_y = np.zeros_like(y)
    for i, lx0 in enumerate(lx):
        mask = np.abs(lx - lx0) < window_width_log
        if np.any(mask):
            smooth_y[i] = np.mean(y[mask])
        else:
            smooth_y[i] = y[i]
    return smooth_y

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Get input phi
# planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_phi1_seed1001_v2.alm
input_klm = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax4096.fits'),(l*(l+1))/2)
input_klm = utils.reduce_lmax(input_klm,lmax=lmax)
input_klm_map = hp.alm2map(input_klm,nside)
#input_clkk = hp.alm2cl(input_klm,input_klm,lmax=lmax)
input_clkk = hp.anafast(input_klm_map * mask,input_klm_map * mask, lmax=lmax) / fsky
print('input_clkk: ',input_clkk)

clkk_auto = np.zeros((10,len(l)),dtype=np.complex_)
clkk_auto_standard = np.zeros((10,len(l)),dtype=np.complex_)
clkk_cross = np.zeros((10,len(l)),dtype=np.complex_)
clkk_cross_standard = np.zeros((10,len(l)),dtype=np.complex_)
for i,idx in enumerate(np.arange(10)+5001):
    print(idx)
    # Get reconstructed 2019/2020 analysis PROFILE-HARDENED phi tracer
    klm = btmp.get_debiased_klm(idx)
    klm = utils.reduce_lmax(klm, lmax=lmax)
    klm_map = hp.alm2map(klm, nside)
    #clkk_auto += hp.anafast(klm_map * mask, lmax=lmax)/fsky

    # Get reconstructed 2019/2020 analysis STANDARD phi tracer
    klm_standard = btmp_standard.get_debiased_klm(idx)
    klm_standard = utils.reduce_lmax(klm_standard, lmax=lmax)
    klm_standard_map = hp.alm2map(klm_standard, nside)
    #clkk_auto_standard += hp.anafast(klm_standard_map * mask, lmax=lmax)/fsky

    # Cross spectrum of reconstructed phi and input phi
    #clkk_cross = hp.alm2cl(klm,input_klm,lmax=lmax)
    clkk_cross[i,:] = hp.anafast(klm_map * mask,input_klm_map * mask, lmax=lmax) / fsky
    clkk_cross_standard[i,:] = hp.anafast(klm_standard_map * mask, input_klm_map * mask, lmax=lmax) / fsky
print('averaging')
clkk_auto_avg = np.mean(clkk_auto,axis=0)
clkk_auto_standard_avg = np.mean(clkk_auto_standard,axis=0)
clkk_cross_avg = np.mean(clkk_cross,axis=0)
clkk_cross_standard_avg = np.mean(clkk_cross_standard,axis=0)

# Plot
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l, moving_average((clkk_cross[0,:])/np.array(input_clkk), window_size=50), color='firebrick',alpha=1,linestyle='-',label='Profile Hardened')
#plt.plot(l, moving_average((clkk_cross_standard[0,:])/np.array(input_clkk), window_size=50), color='darkblue',alpha=1,linestyle='-',label='Standard')
plt.plot(l, moving_average_logx(l,(clkk_cross[0,:])/np.array(input_clkk), window_width_log=0.05), color='firebrick',alpha=1,linestyle='-',label='Profile Hardened')
plt.plot(l, moving_average_logx(l,(clkk_cross_standard[0,:])/np.array(input_clkk), window_width_log=0.05), color='darkblue',alpha=1,linestyle='-',label='Standard')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$L$')
plt.title(r'$C_\ell^{\kappa,in\times\kappa,recon}$ / $C_\ell^{\kappa,in\times\kappa,in}$, Agora Sim 5001',pad=10)
plt.legend(loc='lower left', fontsize='medium')
plt.xscale('log')
plt.ylim(0.6,1.05)
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig('figs/temp.png',bbox_inches='tight')
'''

