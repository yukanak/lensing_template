import subprocess, os, sys
import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy import constants as const
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

nside = 2048
os.environ["HEALPIX"] = "/home/users/yukanaka/miniconda3/envs/tora_py3/Healpix_3.83/"
spice = "/home/users/yukanaka/PolSpice_v03-08-03/bin/spice"

'''
# clii
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/cib_evenring.hpx.fits",
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/mask_apod.hpx.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/cib_oddring.hpx.fits",
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/mask_apod.hpx.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clii.dat",
        "-nlmax",
        "2048",
        "-apodizesigma",
        "30",
        "-thetamax",
        "60",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
        "-decouple",
        "YES",
    ])

# clkk
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4PP_nside2048.fits",
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits",
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clkk_pr4.dat",
        "-nlmax",
        "2048",
        "-apodizesigma",
        "30",
        "-thetamax",
        "60",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
        "-decouple",
        "YES",
    ])

# clik
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits",
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/cib_fullmission.hpx.fits",
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/mask_apod.hpx.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clik_pr4.dat",
        "-nlmax",
        "2048",
        "-apodizesigma",
        "30",
        "-thetamax",
        "60",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
        "-decouple",
        "YES",
    ])
'''

# check
ell = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clii.dat")[:1501,0]
clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clii.dat")[:1501,1] * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clkk_pr4.dat")[:1501,1]
clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clik_pr4.dat")[:1501,1] * (1e6/58.04) # MJy/sr -> uK_CMB
# cib spectra from BKSPT paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0) * (1e6/58.04) # MJy/sr -> uK_CMB
clii_bkspt = np.nanmean(cib['cibauto'], axis=0) * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk_bkspt = cib['clkk']
# Columns are ell,353x545,353x857,545x857,353x353,545x545,857x857,d353x545,d353x857,d545x857,d353x353,d545x545,d857x857
clii_loaded_dat = np.genfromtxt('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/Cl_CIB_T1T2.csv',delimiter=',',comments="#",skip_header=6)
clii_loaded_bin_centers = clii_loaded_dat[:,0]
clii_loaded_545x545 = clii_loaded_dat[:,5] * (1e6/58.04)**2 # MJy/sr -> uK_CMB
# Note that this is PR3 kappa
clik_loaded_dat = np.genfromtxt('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/CIBxKappa_powerspectra.csv',delimiter=',',comments="#",skip_header=5)
clik_loaded = clik_loaded_dat[:,1] * (1e6/58.04) # MJy/sr -> uK_CMB
# Also load theory to get rho
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,1500)
clkk_theory = slpp * (np.arange(1501)*(np.arange(1501)+1))**2/4

# smooth
fwhm_dell = 40 # smoothing width in multipoles (FWHM)
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma
clik_bkspt = gaussian_filter1d(clik_bkspt, sigma=sigma, mode='nearest')
clii_bkspt = gaussian_filter1d(clii_bkspt, sigma=sigma, mode='nearest')
clkk_bkspt = gaussian_filter1d(clkk_bkspt, sigma=sigma, mode='nearest')
clik = gaussian_filter1d(clik, sigma=sigma, mode='nearest')
clii = gaussian_filter1d(clii, sigma=sigma, mode='nearest')
clkk = gaussian_filter1d(clkk, sigma=sigma, mode='nearest')

# BIN
centers = clii_loaded_bin_centers
edges = np.zeros(len(centers) + 1)
edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
edges[0] = centers[0] - (edges[1] - centers[0])
edges[-1] = centers[-1] + (centers[-1] - edges[-2])
which_bin = np.digitize(np.arange(1501), edges) - 1 # bin index per ell
clkk_bkspt_binned = np.array([clkk_bkspt[which_bin == i].mean() for i in range(len(centers))])
clkk_theory_binned = np.array([clkk_theory[which_bin == i].mean() for i in range(len(centers))])

# rho
rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
rho = clik / np.sqrt(clii * clkk_theory)
rho_bkspt_binned = clik_loaded / np.sqrt(clii_loaded_545x545 * clkk_bkspt_binned)
rho_loaded = clik_loaded / np.sqrt(clii_loaded_545x545 * clkk_theory_binned)
print('rho_bkspt: ', rho_bkspt)
print('rho: ', rho)

# plot
plt.figure(0)
plt.clf()

plt.plot(np.arange(1501), clik_bkspt, color='firebrick', alpha=0.5, label='clik BKSPT')
plt.plot(np.arange(1501), clkk_bkspt, color='darkblue', alpha=0.5, label='clkk BKSPT')
plt.plot(np.arange(1501), clii_bkspt, color='darkorange', alpha=0.5, label='clii BKSPT')
plt.plot(np.arange(1501), clik[:1501], color='lightcoral', alpha=0.5, linestyle='--', label='clik')
plt.plot(np.arange(1501), clkk[:1501], color='cornflowerblue', alpha=0.5, linestyle='--', label='clkk')
plt.plot(np.arange(1501), clii[:1501], color='orange', alpha=0.5, linestyle='--', label='clii')
plt.yscale('log')

#plt.plot(np.arange(1501), rho_bkspt[:1501], color='firebrick', alpha=0.5, label='rho BKSPT')
#plt.plot(np.arange(1501), rho[:1501], color='lightcoral', alpha=0.5, label='rho')
#plt.plot(clii_loaded_bin_centers, rho_bkspt_binned, color='thistle', alpha=1, label='rho for Lenz et al. loaded clii & clik + BKSPT clkk')
#plt.plot(clii_loaded_bin_centers, rho_loaded, color='pink', alpha=0.5, label='rho for Lenz et al. loaded clii & clik + theory clkk')

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,1500)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/lenz_et_al_cib_pr4_kappa_specs.png')
