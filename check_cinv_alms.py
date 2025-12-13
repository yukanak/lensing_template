import sys, os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import yaml
sys.path.insert(0, '/home/users/yukanaka/spt3g_software/scratch/wlwu/cinv_lowell/')
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
sys.path.append("/home/users/yukanaka/spt3g_software/scratch/yomori/utils/")
sys.path.insert(0, '/home/users/yukanaka/healqest/pipeline/')
import utils as base_utils
import healqest_utils as utils
import cinv_lowell as cll

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

elm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_5001_elm.fits')
blm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_5001_blm.fits')
elm_data = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_0000_elm.fits')
blm_data = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_0000_blm.fits')

# Expectation
params = yaml.safe_load(open("/home/users/yukanaka/lensing_template/lowell_v3mocks_musebeamv41.yaml"))
lmax = params['lmax']
ell = np.arange(lmax+1)
dir_tmp = params['outdir']
fsky = cll.get_fsky()
nl2d_ee_o = np.load(dir_tmp + params['noise']['nl2d_smooth']['fname']%"ee")/fsky
nl2d_bb_o = np.load(dir_tmp + params['noise']['nl2d_smooth']['fname']%"bb")/fsky
nl1d_ee = hp.alm2cl(np.sqrt(nl2d_ee_o.astype(np.complex_)))
nl1d_bb = hp.alm2cl(np.sqrt(nl2d_bb_o.astype(np.complex_)))
tf1d = cll.get_inv_nvar_weighted_1dtfbeam(params, freqs=[90,150])
clfname = "/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat"
cl_len = utils.get_lensedcls(clfname,lmax=lmax,dict=True)
wf_ee = tf1d**2 * cl_len["ee"] / (tf1d**2 * cl_len["ee"] + nl1d_ee)
wf_bb = tf1d**2 * cl_len["bb"] / (tf1d**2 * cl_len["bb"] + nl1d_bb)
clee_expectation = cl_len["ee"] * wf_ee
clbb_expectation = cl_len["bb"] * wf_bb

clee = hp.alm2cl(hp.almxfl(elm,cl_len["ee"]))/fsky
clbb = hp.alm2cl(hp.almxfl(blm,cl_len["bb"]))/fsky
clee_data = hp.alm2cl(hp.almxfl(elm_data,cl_len["ee"]))/fsky
clbb_data = hp.alm2cl(hp.almxfl(blm_data,cl_len["bb"]))/fsky

idxs = np.arange(499)+1
clee_avg = 0
clbb_avg = 0
for i in idxs:
    elm_i = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_{i:04d}_elm.fits')
    blm_i = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_{i:04d}_blm.fits')
    clee_avg += hp.alm2cl(hp.almxfl(elm_i,cl_len["ee"]))/fsky
    clbb_avg += hp.alm2cl(hp.almxfl(blm_i,cl_len["bb"]))/fsky
clee_avg /= len(idxs)
clbb_avg /= len(idxs)

idxs = np.arange(10)+5001
clee_agora_avg = 0
clbb_agora_avg = 0
for i in idxs:
    elm_i = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_{i:04d}_elm.fits')
    blm_i = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_{i:04d}_blm.fits')
    clee_agora_avg += hp.alm2cl(hp.almxfl(elm_i,cl_len["ee"]))/fsky
    clbb_agora_avg += hp.alm2cl(hp.almxfl(blm_i,cl_len["bb"]))/fsky
clee_agora_avg /= len(idxs)
clbb_agora_avg /= len(idxs)

# Plot
plt.figure(0)
plt.clf()
plt.plot(ell, clee, color='pink', linestyle='-', label='spec of (cinv elm * clee) / fsky, Agora sim 5001')
plt.plot(ell, clee_avg, color='forestgreen', linestyle='-', label='spec of (cinv elm * clee) / fsky, Gaussian sims, averaged')
plt.plot(ell, clee_agora_avg, color='firebrick', linestyle='-', label='spec of (cinv elm * clee) / fsky, Agora sims, averaged')
plt.plot(ell, clee_data, color='darkblue', linestyle='-', label='spec of (cinv elm * clee) / fsky, data')
plt.plot(ell, clee_expectation, color='goldenrod', linestyle='--', label='expectation (wiener filter * clee)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
#plt.ylim(1e-5,1e5)
plt.legend(loc='lower left', fontsize='x-small')
plt.title(f'Cinv Input Spectra')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig('figs/cinv_raw_spec.png',bbox_inches='tight')

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(ell, moving_average(clee_avg/clee_data, window_size=40), color='forestgreen', linestyle='-', alpha=0.8, label='input cinv clee ratio, average of Gaussian sims / data')
plt.plot(ell, moving_average(clee_agora_avg/clee_data, window_size=40), color='firebrick', linestyle='-', alpha=0.8, label='input cinv clee ratio, average of Agora sims / data')
plt.plot(ell, moving_average(clee_agora_avg/clee_avg, window_size=40), color='darkorange', linestyle='-', alpha=0.8, label='input cinv clee ratio, average of Agora sims / Gaussian sims')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0.8,1.3)
plt.legend(loc='upper right', fontsize='x-small')
plt.title(f'Cinv Input Spec Check')
plt.xlabel('$\ell$')
plt.savefig('figs/cinv_spec_ratio.png',bbox_inches='tight')
