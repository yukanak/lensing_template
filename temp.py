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

nside = 2048
lmax = 2000
l = np.arange(lmax+1)
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk = slpp * (l*(l+1))**2/4
mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits")
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
btmp = bt.btemplate(yaml_file,combined_tracer=True)
# CIB map
klm2_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits")
rot = hp.Rotator(coord=['G','C'])
klm2_map = rot.rotate_map_pixel(klm2_map)
klm2_map *= mask * 1e6 / 58.04
klm2 = hp.map2alm(klm2_map, lmax=lmax)
# SPT3G QE
klm1 = btmp.get_debiased_klm(0)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)
# auto and cross
clii1 = hp.alm2cl(klm1)
clii2 = hp.alm2cl(klm2)
clii12 = hp.alm2cl(klm1,klm2)
# does it match what I had in sims?
klm2_map = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_tracers/pr3_cib_pr4_kappa_tracer/cib_tracer_seed1.fits')
klm2_map *= mask
klm2 = hp.map2alm(klm2_map, lmax=lmax)
klm1 = btmp.get_debiased_klm(1)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)
clii1_sim1 = hp.alm2cl(klm1)
clii2_sim1 = hp.alm2cl(klm2)
clii12_sim1 = hp.alm2cl(klm1,klm2)

rho = clii12/np.sqrt(clii1*clii2)
print('rho, data: ',np.mean(rho[50:200]))
rho = clii12_sim1/np.sqrt(clii1_sim1*clii2_sim1)
print('rho, sim1: ',np.mean(rho[50:200]))

#=============================================================================#
fsky = np.mean(mask**2)
# try crossing Agora CIB realization at 545 GHz with Agora GMV reconstructed kappa
agora_cib_map = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_len_mag_cibmap_planck_545ghz_nside2048.fits')
agora_cib_map *= mask * 1 / 58.04
klm1 = btmp.get_debiased_klm(5001)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)
klm1_map = hp.alm2map(klm1,nside)
#agora_cib_cross_recon_qe = hp.alm2cl(klm1,agora_cib) * np.mean(mask) # FSKY (but this makes no sense?)
agora_cib_cross_recon_qe = hp.anafast(agora_cib_map,klm1_map)/fsky
# INPUT KAPPA FOR AGORA
klm_input_agora = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax4096.fits'),(np.arange(4096+1)*(np.arange(4096+1)+1))/2)
#klm_input_agora = utils.reduce_lmax(klm_input_agora, lmax=lmax)
klm_input_agora_map = hp.alm2map(klm_input_agora,nside)
#hp.write_map(f"/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_kappa_map_nside{nside}.fits", klm_input_agora_map)
klm_input_agora_map *= mask
#agora_cib_cross_input_klm = hp.alm2cl(klm_input_agora,agora_cib) * np.mean(mask**2)**2
#agora_recon_qe_cross_input_klm = hp.alm2cl(klm_input_agora,klm1) / np.mean(mask)
agora_cib_cross_input_klm = hp.anafast(agora_cib_map,klm_input_agora_map)/fsky
agora_recon_qe_cross_input_klm = hp.anafast(klm1_map,klm_input_agora_map)/fsky
agora_input_klm = hp.anafast(klm_input_agora_map,klm_input_agora_map)/fsky
agora_cib = hp.anafast(agora_cib_map,agora_cib_map)/fsky
# BKSPT clii
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0) * (1e6/58.04) # MJy/sr -> uK_CMB
clii_bkspt = np.nanmean(cib['cibauto'], axis=0) * (1e6/58.04)**2 # MJy/sr -> uK_CMB
#=============================================================================#

# plot
plt.figure(0)
plt.clf()
plt.plot(l, clkk, 'k', label='Fiducial $C_L^{\kappa\kappa}$')
plt.plot(l, agora_cib_cross_recon_qe[:lmax+1], color='lightcoral', alpha=0.8, label='agora cib x agora recon qe')
plt.plot(l, agora_cib_cross_input_klm[:lmax+1], color='lightgreen', alpha=0.8, label='agora cib x agora input klm')
plt.plot(l, agora_recon_qe_cross_input_klm[:lmax+1], color='slateblue', alpha=0.8, label='agora input klm x agora recon qe')
plt.plot(l, agora_input_klm[:lmax+1], color='orange', alpha=0.8, label='agora input klm auto')
plt.plot(l, agora_cib[:lmax+1], color='magenta', alpha=0.8, label='agora cib auto')
plt.plot(l[:1501], clii_bkspt[:1501], color='violet', linestyle='--', alpha=0.8, label='BKSPT clii')
plt.plot(l[:1501], clik_bkspt[:1501], color='silver', linestyle='--', alpha=0.8, label='BKSPT clik')
plt.yscale('log')
#plt.axhline(y=1, color='gray', linestyle='--')
#plt.plot(l, agora_cib_cross_recon_qe/clkk, color='lightcoral', alpha=0.8, label='agora cib x agora recon qe / fid clkk')
#plt.plot(l, agora_cib_cross_input_klm/clkk, color='lightgreen', alpha=0.8, label='agora cib x agora input klm / fid clkk')
#plt.plot(l, agora_recon_qe_cross_input_klm/clkk, color='slateblue', alpha=0.8, label='agora input klm x agora recon qe / fid clkk')
#plt.plot(l, clii1/clii1_sim1, color='lightgreen', alpha=0.8, label='clii1 data / sim 1')
#plt.plot(l, clii2/clii2_sim1, color='lightcoral', alpha=0.8, label='clii2 data / sim 1')
#plt.plot(l, clii12/clii12_sim1, color='slateblue', alpha=0.8, label='clii12 data / sim 1')
#plt.ylim(0.5,1.5)
#plt.ylim(-1,2)
#plt.ylim(0,0.3)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='small')
plt.xscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')


