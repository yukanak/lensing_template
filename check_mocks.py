import numpy as np
import sys, os
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
sys.path.append('/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/yuuki_scripts/')
import utils as yuuki_utils

# check mocks, reconstructed clkk
l = np.arange(0,4096+1)
ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
slpp = np.insert(slpp, 0, 0)
slpp = np.insert(slpp, 0, 0)
clkk = slpp[:4097] * (l*(l+1))**2/4
lmax = 4096
bins = np.logspace(np.log10(30), np.log10(4000), 51)
digitized = np.digitize(np.arange(4001), bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

'''
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

'''
#=============================================================================#
# check mocks, plms
yaml_file = 'bt_gmv3500.yaml'
yaml_file_prfhrd = 'bt_gmv3500_prfhrd.yaml'
yaml_file_agoraGfgs = 'bt_gmv3500_agoraGfgs.yaml'
yaml_file_agoraGfgs_prfhrd = 'bt_gmv3500_agoraGfgs_prfhrd.yaml'
btmp = bt.btemplate(yaml_file)
btmp_prfhrd = bt.btemplate(yaml_file_prfhrd)
btmp_agoraGfgs = bt.btemplate(yaml_file_agoraGfgs)
btmp_agoraGfgs_prfhrd = bt.btemplate(yaml_file_agoraGfgs_prfhrd)
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size
nside = btmp.nside
klm_fid = btmp.get_debiased_klm(5001)
klm_fid_prfhrd = btmp_prfhrd.get_debiased_klm(5001)
klm_agoraGfgs = btmp_agoraGfgs.get_debiased_klm(5001)
klm_agoraGfgs_prfhrd = btmp_agoraGfgs_prfhrd.get_debiased_klm(5001)
kmap_fid = hp.alm2map(klm_fid, nside)
kmap_fid_prfhrd = hp.alm2map(klm_fid_prfhrd, nside)
kmap_agoraGfgs = hp.alm2map(klm_agoraGfgs, nside)
kmap_agoraGfgs_prfhrd = hp.alm2map(klm_agoraGfgs_prfhrd, nside)
auto_fid = hp.anafast(kmap_fid * mask) / fsky
auto_fid_prfhrd = hp.anafast(kmap_fid_prfhrd * mask) / fsky
auto_agoraGfgs = hp.anafast(kmap_agoraGfgs * mask) / fsky
auto_agoraGfgs_prfhrd = hp.anafast(kmap_agoraGfgs_prfhrd * mask) / fsky
lbins = np.logspace(np.log10(30), np.log10(2000), 20)
ldigitized = np.digitize(np.arange(2001), lbins)
lbin_centers = (lbins[:-1] + lbins[1:]) / 2
gaussian = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/clkk_polspice_mfxxonly_mfsplit_nops/cls_kgmvbhttprf_nsims1_498_nsims2_248_mcresp_mfxxonly_mfsplit_didx_0_with_subinput_debias_SIMN0.npz',allow_pickle=True)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(lbin_centers,np.array([(auto_fid_prfhrd[:2001]/auto_agoraGfgs_prfhrd[:2001])[ldigitized == i].mean() for i in range(1, len(lbins))]),label='sim 5001 prfhrd gmvjtp Agora fid klm auto / G fgs klm auto')
plt.plot(lbin_centers,gaussian['dvec0'].item()['rcl']/np.array([(auto_agoraGfgs_prfhrd[:2001])[ldigitized == i].mean() for i in range(1, len(lbins))]),label='sim mean prfhrd gmvjtp Gaussian klm auto / sim 5001 Agora G fgs klm auto')
plt.plot(lbin_centers,np.array([(auto_fid[:2001]/auto_agoraGfgs[:2001])[ldigitized == i].mean() for i in range(1, len(lbins))]),label='sim 5001 gmvjtp Agora fid klm auto / G fgs klm auto')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,2000)
plt.ylim(0.9,1.2)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

plt.clf()
plt.plot(l[:2001],clkk[:2001],color='k',label='theory clkk')
plt.plot(l[:2001],auto_fid[:2001],color='firebrick',label='sim 5001 gmvjtp klm')
plt.plot(l[:2001],auto_fid_prfhrd[:2001],color='forestgreen',label='sim 5001 prfhrd gmvjtp klm')
plt.plot(l[:2001],auto_agoraGfgs[:2001],color='salmon',linestyle='--',alpha=0.5,label='sim 5001 gmvjtp klm, G fgs')
plt.plot(l[:2001],auto_agoraGfgs_prfhrd[:2001],color='lightgreen',linestyle='--',alpha=0.5,label='sim 5001 prfhrd gmvjtp klm, G fgs')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.ylim(5e-8,4e-7)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

'''
#=============================================================================#
# check mocks, after coadd_one
mock_new = '/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/outputs/healpix/seed5001/Coadd_allfields_150ghz.fits'
mock_agora = '/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/yuuki_scripts/seed5001/Coadd_allfields_150ghz.fits'
# for pol (Q/U) foregrounds should be low, so difference and compare with Yuuki's mock_agora which has foregrounds
t,q,u = yuuki_utils.read_partialmap(mock_new, scal=1, pol=True, flipU=False)
t_yuuki,q_yuuki,u_yuuki = yuuki_utils.read_partialmap(mock_agora, scal=1, pol=True, flipU=False)
diff = q - q_yuuki

plt.figure(0)
plt.clf()
plt.plot(diff)
plt.axhline(y=0, color='gray', linestyle='--')
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/check_mocks.png')

#=============================================================================#
# check mocks, after apply_weights_notfdeconv
# before cinv
mock_new = '/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/ilc/cmb/data/data_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_seed5001_withsignflipnoise_2dilc_crosstf_full_052425_notf_agoraGfgs.npz'
mock_G = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/xilc_inputs_precinv/data_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_seed5001_withsignflipnoise_2dilc_crosstf_full_052425_notf.npz'

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
#plt.yscale('log')
plt.plot(bin_centers, np.array([(cltt[:4001]/cltt_G[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='firebrick', alpha=0.5, label='cltt new mock sim 5001 w G fgs / sim 5001 normal')
plt.plot(bin_centers, np.array([(clee[:4001]/clee_G[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='darkblue', alpha=0.5, label='clee new mock sim 5001 w G fgs / sim 5001 normal')
plt.axhline(y=1, color='gray', linestyle='--')
plt.ylim(0.95,1.07)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,4000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/check_mocks.png')

#=============================================================================#
# check mocks, after cinv, jTP
# 5001
fn_yuuki = '/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/cinv_jtp_cmbmv_v5_aggcinv_crosstf_measninvps_optimizedinp/tqu1_nside2048_lmin350_lmax4096_mmin100_gmv052425_seed5001_ananinv_v3_highninv_2dcinv_binmaskcinv_v3_notf_Ttol5e-4_v4.npz'
fn_yuka = '/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch_gaussian_fgs/cinv_jtp_cmbmv_v5_aggcinv_crosstf_measninvps_optimizedinp/tqu1_nside2048_lmin350_lmax4096_mmin100_gmv052425_seed5001_ananinv_v3_highninv_2dcinv_binmaskcinv_v3_notf_Ttol5e-4_v4.npz'
d_yuuki = np.load(fn_yuuki)
d_yuka = np.load(fn_yuka)
tlm1,elm1,blm1 = d_yuuki['tlm'],d_yuuki['elm'],d_yuuki['blm']
tlm4,elm4,blm4 = d_yuka['tlm'],d_yuka['elm'],d_yuka['blm']
cltt1 = hp.alm2cl(tlm1)
cltt4 = hp.alm2cl(tlm4)
clee1 = hp.alm2cl(elm1)
clee4 = hp.alm2cl(elm4)
clbb1 = hp.alm2cl(blm1)
clbb4 = hp.alm2cl(blm4)

plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(bin_centers, np.array([(cltt1[:4001]/cltt4[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='darkblue', linestyle='-', alpha=0.8, label='sim 5001 input cinv cltt ratio, normal / w G fgs')
plt.plot(bin_centers, np.array([(clee1[:4001]/clee4[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='pink', linestyle='-', alpha=0.8, label='sim 5001 input cinv clee ratio, normal / w G fgs')
plt.xscale('log')
plt.xlim(200,4096)
#plt.ylim(0.95,1.05)
plt.legend(loc='upper right')
plt.title(f'Cinv Input Spec Check')
plt.xlabel('$\ell$')
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png',bbox_inches='tight')

# G 1-10
idxs = np.arange(10)+1
cltt = 0
clee = 0
cltt4 = 0
clee4 = 0
for idx in idxs:
    #fn_yuuki = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/cinv_jtp_cmbmv_v5_aggcinv_crosstf_measninvps_optimizedinp/tqu1_nside2048_lmin350_lmax4096_mmin100_gmv052425_seed{idx}_ananinv_v3_highninv_2dcinv_binmaskcinv_v3_notf_Ttol5e-4_v4.npz'
    # below is actually my reprocessed ver, NOT yuuki's
    fn_yuuki = f'/scratch/users/yukanaka/cinv_jtp_cmbmv_v5_aggcinv_crosstf_measninvps_optimizedinp/tqu1_nside2048_lmin350_lmax4096_mmin100_gmv052425_seed{idx}_ananinv_v3_highninv_2dcinv_binmaskcinv_v3_notf_Ttol5e-4_v4.npz'
    fn_yuka = f'/oak/stanford/orgs/kipac/users/yukanaka/gaussian_input_skies_spt3g_patch/cinv_jtp_cmbmv_v5_aggcinv_crosstf_measninvps_optimizedinp/tqu1_nside2048_lmin350_lmax4096_mmin100_gmv052425_seed{idx}_ananinv_v3_highninv_2dcinv_binmaskcinv_v3_notf_Ttol5e-4_v4.npz'
    d_yuuki = np.load(fn_yuuki)
    d_yuka = np.load(fn_yuka)
    tlm1,elm1,blm1 = d_yuuki['tlm'],d_yuuki['elm'],d_yuuki['blm']
    tlm4,elm4,blm4 = d_yuka['tlm'],d_yuka['elm'],d_yuka['blm']
    cltt4 += hp.alm2cl(tlm4)
    clee4 += hp.alm2cl(elm4)
    # CROSS, plot ratio of cross with auto of noiseless, should be 1
    cltt += hp.alm2cl(tlm1,tlm4)
    clee += hp.alm2cl(elm1,elm4)
cltt4 /= 10
clee4 /= 10
cltt /= 10
clee /= 10

plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(bin_centers, np.array([(cltt[:4001]/cltt4[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='darkblue', linestyle='-', alpha=0.8, label='avg sims 1-10 cinv cltt ratio, normal x noiseless / noiseless')
plt.plot(bin_centers, np.array([(clee[:4001]/clee4[:4001])[digitized == i].mean() for i in range(1, len(bins))]), color='pink', linestyle='-', alpha=0.8, label='avg sims 1-10 cinv clee ratio, normal x noiseless / noiseless')
plt.xscale('log')
plt.xlim(200,4096)
#plt.ylim(0.95,1.05)
plt.legend(loc='upper right')
plt.title(f'Cinv Spec Check')
plt.xlabel('$\ell$')
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png',bbox_inches='tight')
'''


