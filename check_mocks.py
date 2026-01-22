import numpy as np
import sys, os
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

# check mocks, reconstructed clkk
l = np.arange(0,4096+1)
ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
slpp = np.insert(slpp, 0, 0)
slpp = np.insert(slpp, 0, 0)
clkk = slpp[:4097] * (l*(l+1))**2/4
lmax = 4096
# Gaussian
gaussian = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_with_subinput_debias.npz',allow_pickle=True)
gaussian_ratio = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/ratio_cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_with_subinput_debias.npz',allow_pickle=True)
ratio_agora_over_agoraGfgs = np.zeros((10,len(gaussian['dvec0'].item()['rl'])))
mean_agora = 0
mean_agoraGfgs = 0
idxs = np.arange(10)+5001
for i,idx in enumerate(idxs):
    # NEW AGORA SIMS WITH G FGS
    agoraGfgs = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4//clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmv_nsims1_488_nsims2_238_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_agoraGfgs.npz',allow_pickle=True)
    agoraGfgs_ratio = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4//clkk_polspice_mfxxonly_mfsplit_nops/ratio_cls_kgmv_nsims1_488_nsims2_238_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_agoraGfgs.npz',allow_pickle=True)
    # NEW AGORA SIMS WITH G FGS, USING SIM N0
    agoraGfgs_SIMN0 = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4//clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmv_nsims1_488_nsims2_238_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_agoraGfgs_SIMN0.npz',allow_pickle=True)
    agoraGfgs_SIMN0_ratio = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4//clkk_polspice_mfxxonly_mfsplit_nops/ratio_cls_kgmv_nsims1_488_nsims2_238_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_agoraGfgs_SIMN0.npz',allow_pickle=True)
    # AGORA
    agora = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias.npz',allow_pickle=True)
    agora_ratio = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/ratio_cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias.npz',allow_pickle=True)
    # AGORA, USING SIM N0
    agora_SIMN0 = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4//clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_SIMN0.npz',allow_pickle=True)
    agora_SIMN0_ratio = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/ratio_cls_kgmv_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_didx_{idx}_with_subinput_debias_SIMN0.npz',allow_pickle=True)
    #ratio_agora_over_agoraGfgs[i,:] = agora['dvec0'].item()['rdl_corr']/agoraGfgs['dvec0'].item()['rdl_corr']
    #mean_agora += agora['dvec0'].item()['rdl_corr']
    #mean_agoraGfgs += agoraGfgs['dvec0'].item()['rdl_corr']
    ratio_agora_over_agoraGfgs[i,:] = agora_SIMN0['dvec0'].item()['rdl_corr']/agoraGfgs_SIMN0['dvec0'].item()['rdl_corr']
    mean_agora += agora_SIMN0['dvec0'].item()['rdl_corr']
    mean_agoraGfgs += agoraGfgs_SIMN0['dvec0'].item()['rdl_corr']
mean_agora /= 10
mean_agoraGfgs /= 10
ratio_agora_over_gaussian = mean_agora / gaussian['dvec0'].item()['rcl']
ratio_agoraGfgs_over_gaussian = mean_agoraGfgs / gaussian['dvec0'].item()['rcl']

# RDN0 debug
dir_cls = '/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/clkk_polspice_mfxxonly_mfsplit_nops/'
rdn0_firsthalf = utils.loadcls(dir_cls,244,'gmv','RDN0',N0=agoraGfgs['N0'],Lmin=24,Lmax=2500,use_cache=True,verbose=False,didx=5001,startidx=11)
rdn0_secondhalf = utils.loadcls(dir_cls,244,'gmv','RDN0',N0=agoraGfgs['N0'],Lmin=24,Lmax=2500,use_cache=True,verbose=False,didx=5001,startidx=255)

# Plot
plt.figure(0)
plt.clf()
#plt.plot(l, clkk, 'gray', label='Fiducial $C_L^{\kappa\kappa}$')
#plt.plot(gaussian['dvec0'].item()['rl'],gaussian['dvec0'].item()['rcl'], color='firebrick', linestyle='-', alpha=0.8, label='Binned Sim Mean, Gaussian')
#plt.plot(gaussian['dvec0'].item()['rl'],gaussian['dvec0'].item()['rdl_corr'], color='pink', linestyle='--', alpha=0.8, label='Binned Data, Gaussian')
#plt.plot(agoraGfgs['dvec0'].item()['rl'],agoraGfgs['dvec0'].item()['rdl_corr'], color='cornflowerblue', linestyle='--', alpha=0.8, label=f'Binned Data ({idx}), Agora WITH G FGS')
#plt.plot(agora['dvec0'].item()['rl'],agora['dvec0'].item()['rdl_corr'], color='lightgreen', linestyle='--', alpha=0.8, label=f'Binned Data ({idx}), Agora')
#plt.ylabel("$C_L^{\kappa\kappa}$")
#plt.yscale('log')
#plt.ylim(1e-9,1e-6)
#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l[:4001],agoraGfgs['RDN0']/gaussian['N0'], color='darkblue', linestyle='-', alpha=0.8, label='RDN0 Agora WITH G FGS / N0 Gaussian')
#plt.plot(l[:4001],agora['RDN0']/gaussian['N0'], color='forestgreen', linestyle='-', alpha=0.8, label='RDN0 Agora / N0 Gaussian')
#plt.plot(l[:4001],agoraGfgs['N0']/gaussian['N0'], color='magenta', linestyle='-', alpha=0.8, label='N0 Agora WITH G FGS / N0 Gaussian')
#plt.plot(l[:4001],rdn0_firsthalf/rdn0_secondhalf, color='firebrick', linestyle='-', alpha=0.8, label='RDN0 Agora WITH G FGS idx 11-254 / idx 255-498')
#plt.ylim(0.5,1.5)
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_gaussian,color='firebrick',label='mean(Agora)/mean(Gaussian)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agoraGfgs_over_gaussian,color='darkblue',label='mean(Agora with G fgs)/mean(Gaussian)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[0,:],color='silver',alpha=0.8,label='Agora/Agora with G fgs')#label='Agora/Agora with G fgs (rlz 5001)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[1,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5001)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[2,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5002)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[3,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5003)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[4,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5004)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[5,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5005)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[6,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5006)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[7,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5007)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[8,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5008)')
plt.plot(gaussian['dvec0'].item()['rl'],ratio_agora_over_agoraGfgs[9,:],color='silver',alpha=0.8,)#label='Agora/Agora with G fgs (rlz 5009)')
plt.ylim(0.5,1.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$L$')
plt.title(f"Reconstruction Results",pad=10)
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig(f'/home/users/yukanaka/lensing_template/figs/temp.png',bbox_inches='tight')

#=============================================================================#
# check mocks, raw

seed = 5001
mock_new = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/ilc/cmb/data/data_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_seed{seed:04}_withsignflipnoise_2dilc_crosstf_full_052425_notf_agoraGfgs.npz'
seed_G = 1
mock_G = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/xilc_inputs_precinv/data_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_seed{seed_G:04}_withsignflipnoise_2dilc_crosstf_full_052425_notf.npz'

alm_new = np.load(mock_new)
tlm = alm_new['almT']
elm = alm_new['almE']
blm = alm_new['almB']
alm_G = np.load(mock_G)
tlm_G = alm_G['almT']
elm_G = alm_G['almE']
blm_G = alm_G['almB']
cltt = hp.alm2cl(tlm)
clee = hp.alm2cl(elm)
cltt_G = hp.alm2cl(tlm_G)
clee_G = hp.alm2cl(elm_G)

plt.figure(0)
plt.clf()
#plt.plot(np.arange(2001), cltt[:2001], color='firebrick', alpha=0.5, label='cltt new mock sim 5001')
#plt.plot(np.arange(2001), clee[:2001], color='darkblue', alpha=0.5, label='clee new mock sim 5001')
#plt.plot(np.arange(2001), cltt_G[:2001], color='lightcoral', alpha=0.5, label='cltt G sim 1')
#plt.plot(np.arange(2001), clee_G[:2001], color='cornflowerblue', alpha=0.5, linestyle='-', label='clee G sim 1')
#plt.yscale('log')
plt.plot(np.arange(2001), cltt[:2001]/cltt_G[:2001], color='firebrick', alpha=0.5, label='cltt new mock sim 5001 / G sim 1')
plt.plot(np.arange(2001), clee[:2001]/clee_G[:2001], color='darkblue', alpha=0.5, label='clee new mock sim 5001 / G sim 1')
plt.axhline(y=1, color='gray', linestyle='--')
plt.ylim(0,3)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/check_mocks.png')
