import subprocess, os
import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy import constants as const

nside = 2048
os.environ["HEALPIX"] = "/home/users/yukanaka/miniconda3/envs/tora_py3/Healpix_3.83/"
spice = "/home/users/yukanaka/PolSpice_v03-08-03/bin/spice"
# Columns are ell,353x545,353x857,545x857,353x353,545x545,857x857,d353x545,d353x857,d545x857,d353x353,d545x545,d857x857
clii_loaded_dat = np.genfromtxt('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/Cl_CIB_T1T2.csv',delimiter=',',comments="#",skip_header=6)
clii_loaded_bin_centers = clii_loaded_dat[:,0]
clii_loaded_545x545 = clii_loaded_dat[:,5]
# Note that this is PR3 kappa
clik_loaded_dat = np.genfromtxt('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/CIBxKappa_powerspectra.csv',delimiter=',',comments="#",skip_header=5)
clik_loaded = clik_loaded_dat[:,1]
# clkk spectrum from BKSPT paper                                                  
clkk_bkspt = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")['clkk']
fwhm_dell = 40 # smoothing width in multipoles (FWHM)                           
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma            
clkk_bkspt = gaussian_filter1d(clkk_bkspt, sigma=sigma, mode='nearest')
# BIN
centers = clii_loaded_bin_centers
edges = np.zeros(len(centers) + 1)
edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
edges[0] = centers[0] - (edges[1] - centers[0])
edges[-1] = centers[-1] + (centers[-1] - edges[-2])
which_bin = np.digitize(np.arange(1501), edges) - 1 # bin index per ell
clkk_bkspt = np.array([clkk_bkspt[which_bin == i].mean() for i in range(len(centers))])
# clkk from PolSpice using PR3 map
clkk_pr3 = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clkk_pr3.dat")[:1501,1]
clkk_pr3 = gaussian_filter1d(clkk_pr3, sigma=sigma, mode='nearest')
# BIN
clkk_pr3 = np.array([clkk_pr3[which_bin == i].mean() for i in range(len(centers))])
# Look at rho
rho_bkspt = clik_loaded / np.sqrt(clii_loaded_545x545 * clkk_bkspt)
rho_polspice = clik_loaded / np.sqrt(clii_loaded_545x545 * clkk_pr3)
# PLOT
plt.figure(0)                                                                   
plt.clf()                                                                       
plt.plot(clii_loaded_bin_centers, rho_bkspt, color='firebrick', alpha=0.5, label='rho for Lenz et al. loaded clii & clik + BKSPT clkk')
plt.plot(clii_loaded_bin_centers, rho_polspice, color='lightcoral', alpha=0.5, label='rho for Lenz et al. loaded clii & clik + PolSpice PR3 clkk')
plt.grid(True, linestyle="--", alpha=0.5)                                       
plt.xlabel('$\ell$')                                                            
plt.legend(loc='upper right', fontsize='small')                                 
plt.xscale('log')                                                               
plt.xlim(10,1500)                                                               
plt.tight_layout()                                                              
plt.savefig('/home/users/yukanaka/lensing_template/figs/lenz_et_al_loaded_cib_kappa_specs.png')
                                                                                               
#hdu = fits.open("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask.fits.gz")
#m = hdu[1].data.field(0).astype(float).ravel()
#hp.write_map("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits", m, overwrite=True)
#klm = hp.read_alm("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/PR42018like_klm_dat_MV.fits")
#kappa_map = hp.alm2map(klm, nside=nside)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits", kappa_map)
# make common mask
#mask_kappa = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits")
#mask_cib = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits")
#mask_common = np.logical_and(mask_kappa > 0.5, mask_cib > 0.5).astype(float)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common.fits", mask_common, overwrite=True)
#mask_common = mask_kappa * mask_cib
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits", mask_common, overwrite=True)

'''
# clii
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH1.fits",
        "-maskfile",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH2.fits",
        "-maskfile2",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat",
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
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits",
        "-maskfile2",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clkk_pr4.dat",
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
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits",
        "-maskfile2",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clik_545ghz_pr4.dat",
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
ell = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat")[:1501,0]
clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat")[:1501,1]
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clkk_pr4.dat")[:1501,1]
clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clik_545ghz_pr4.dat")[:1501,1]
# cib spectra from BKSPT paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0)
clii_bkspt = np.nanmean(cib['cibauto'], axis=0)
clkk_bkspt = cib['clkk']

# try simple alm2cl
# this is masked!
#cib_fiona_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits")
#cib_fiona_map = np.nan_to_num(cib_fiona_map, nan=0.0)
#cib_fiona_alm = hp.map2alm(cib_fiona_map)
#clii_alm2cl = hp.alm2cl(cib_fiona_alm)
# try simple anafast
cib_fiona_map_rh1 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH1.fits")
cib_fiona_map_rh2 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH2.fits")
kap_pr4_PP = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4PP_nside2048.fits")
kap_pr4_TT = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits")
cib_fiona_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits")
kap_pr4 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits")
cib_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits")
kap_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits")
mask_common = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits")
fsky = np.mean(mask_common**2)
cib_fiona_map_rh1 *= mask_common
cib_fiona_map_rh2 *= mask_common
cib_fiona_map *= mask_common
kap_pr4_PP *= mask_common
kap_pr4_TT *= mask_common
kap_pr4 *= mask_common
cib_fiona_map_rh1 = np.nan_to_num(cib_fiona_map_rh1, nan=0.0)
cib_fiona_map_rh2 = np.nan_to_num(cib_fiona_map_rh2, nan=0.0)
cib_fiona_map = np.nan_to_num(cib_fiona_map, nan=0.0)
#cib_fiona_map_rh1 -= np.mean(cib_fiona_map_rh1[mask_common > 0])                   
#cib_fiona_map_rh2 -= np.mean(cib_fiona_map_rh2[mask_common > 0])                   
#cib_fiona_map -= np.mean(cib_fiona_map[mask_common > 0])                           
#kap_pr4_PP -= np.mean(kap_pr4_PP[mask_common > 0])                                  
#kap_pr4_TT -= np.mean(kap_pr4_TT[mask_common > 0])                                  
#kap_pr4 -= np.mean(kap_pr4[mask_common > 0])
clii_anafast = hp.anafast(cib_fiona_map_rh1,cib_fiona_map_rh2) / fsky
clkk_anafast = hp.anafast(kap_pr4_PP,kap_pr4_TT) / fsky
clik_anafast = hp.anafast(kap_pr4,cib_fiona_map) / fsky

# smooth                                                                        
fwhm_dell = 40 # smoothing width in multipoles (FWHM)
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma
clik_bkspt = gaussian_filter1d(clik_bkspt, sigma=sigma, mode='nearest')
clii_bkspt = gaussian_filter1d(clii_bkspt, sigma=sigma, mode='nearest')
clkk_bkspt = gaussian_filter1d(clkk_bkspt, sigma=sigma, mode='nearest')
clik = gaussian_filter1d(clik, sigma=sigma, mode='nearest')
clii = gaussian_filter1d(clii, sigma=sigma, mode='nearest')
clkk = gaussian_filter1d(clkk, sigma=sigma, mode='nearest')
clik_anafast = gaussian_filter1d(clik_anafast, sigma=sigma, mode='nearest')     
clii_anafast = gaussian_filter1d(clii_anafast, sigma=sigma, mode='nearest')     
clkk_anafast = gaussian_filter1d(clkk_anafast, sigma=sigma, mode='nearest')

# rho
rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
rho = clik / np.sqrt(clii * clkk)
rho_anafast = clik_anafast / np.sqrt(clii_anafast * clkk_anafast)
print('rho_bkspt: ', rho_bkspt)
print('rho: ', rho)
print('rho_anafast: ', rho_anafast)

# plot
plt.figure(0)
plt.clf()

#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(np.arange(1501), clik_bkspt/clik[:1501], color='lightcoral', alpha=0.5, label='clik BKSPT / clik')
#plt.plot(np.arange(1501), clkk_bkspt/clkk[:1501], color='cornflowerblue', alpha=0.5, label='clkk BKSPT / clkk')
#plt.plot(np.arange(1501), clii_bkspt/clii[:1501], color='orange', alpha=0.5, label='clii BKSPT / clii')
#plt.ylim(0,100)
#plt.yscale('log')

#plt.plot(np.arange(1501), clik_bkspt*1e6, color='firebrick', alpha=0.5, label='clik BKSPT')
#plt.plot(np.arange(1501), clkk_bkspt, color='darkblue', alpha=0.5, label='clkk BKSPT')
#plt.plot(np.arange(1501), clii_bkspt*1e12, color='darkorange', alpha=0.5, label='clii BKSPT')
#plt.plot(np.arange(1501), clik[:1501], color='lightcoral', alpha=0.5, linestyle='--', label='clik')
#plt.plot(np.arange(1501), clkk[:1501], color='cornflowerblue', alpha=0.5, linestyle='--', label='clkk')
#plt.plot(np.arange(1501), clii[:1501], color='orange', alpha=0.5, linestyle='--', label='clii')
##plt.plot(np.arange(1501), clii_alm2cl[:1501], color='plum', alpha=0.5, linestyle='--', label='clii from alm2cl')
#plt.yscale('log')

plt.plot(np.arange(1501), rho_bkspt[:1501], color='firebrick', alpha=0.5, label='rho BKSPT')
plt.plot(np.arange(1501), rho[:1501], color='lightcoral', alpha=0.5, label='rho')
plt.plot(np.arange(1501), rho_anafast[:1501], color='thistle', linestyle='--', alpha=1, label='rho from anafast')

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,1500)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/fiona_545ghz_cib_pr4_kappa_specs.png')
