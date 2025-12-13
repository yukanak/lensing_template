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
#hdu = fits.open("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask.fits.gz")
#m = hdu[1].data.field(0).astype(float).ravel()
#hp.write_map("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits", m, overwrite=True)
#klm = hp.read_alm("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/PR42018like_klm_dat_MV.fits")
#kappa_map = hp.alm2map(klm, nside=nside)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits", kappa_map)

# Make common mask
#mask_kappa = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits")
#mask_cib = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits")
#mask_common = np.logical_and(mask_kappa > 0.5, mask_cib > 0.5).astype(float)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common.fits", mask_common, overwrite=True)
#mask_common = mask_kappa * mask_cib
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits", mask_common, overwrite=True)

## Try plotting map
#cib_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits")
#kap_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits")
##m_cib_pr3 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits") # KNOW this is Galactic coord
#m_cib, hdr_cib = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits", h=True)
#m_kappa, hdr_kappa = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits", h=True)
##hp.mollview(m_cib_pr3,title='PR3 CIB in Galactic',coord=['G','G'])
##plt.savefig('/home/users/yukanaka/lensing_template/figs/pr3_cib_map.png')
##hp.mollview(m_cib_pr3,title='PR3 CIB Rotated to Equatorial',coord=['G','C'])
##plt.savefig('/home/users/yukanaka/lensing_template/figs/pr3_cib_map2.png')
#hp.mollview(m_cib*cib_mask,title='Masked Fiona CIB in Galactic',coord=['G','G'])
#plt.savefig('/home/users/yukanaka/lensing_template/figs/fiona_cib_map.png')
##hp.mollview(m_cib,title='Fiona CIB Rotated to Equatorial',coord=['G','C'])
##plt.savefig('/home/users/yukanaka/lensing_template/figs/fiona_cib_map2.png')
#hp.mollview(m_kappa*kap_mask,title='Masked PR4 Kappa in Galactic',coord=['G','G'])
#plt.savefig('/home/users/yukanaka/lensing_template/figs/pr4_kappa_map.png')
##hp.mollview(m_kappa,title='PR4 Kappa Rotated to Equatorial',coord=['G','C'])
##plt.savefig('/home/users/yukanaka/lensing_template/figs/pr4_kappa_map2.png')

'''
# clii
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH1.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits", # PR3
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-beam",
        "5",
        "-pixelfile",
        "YES",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH2.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits", # PR3
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-beam2",
        "5",
        "-pixelfile2",
        "YES",
        "-clfile",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clii_545ghz_fionamask.dat", # PR3
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz_fullmission.dat",
        "-nlmax",
        "2048",
        "-apodizesigma",
        "15",
        "-thetamax",
        "20",
        "-subdipole",
        "YES",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
    ])

# clkk
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4PP_nside2048.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits", # PR3
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits", # PR3
        "-pixelfile",
        "YES",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits", # PR3
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits", # PR3
        "-pixelfile2",
        "YES",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clkk_pr4.dat",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clkk_pr3.dat", # PR3
        "-nlmax",
        "2048",
        "-apodizesigma",
        "15",
        "-thetamax",
        "20",
        "-subdipole",
        "YES",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
    ])

# clik
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits", # PR3
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits", # PR3
        "-pixelfile",
        "YES",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits", # PR3
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits",
        "-beam2",
        "5",
        "-pixelfile2",
        "YES",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clik_545ghz_pr4.dat",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/separate_masks/cls_clik_545ghz_pr4.dat",
        #"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_pr3_fionamask.dat", # PR3
        "-nlmax",
        "2048",
        "-apodizesigma",
        "15",
        "-thetamax",
        "20",
        "-subdipole",
        "YES",
        "-subav",
        "YES",
        "-apodizetype",
        "0",
        "-verbosity",
        "NO",
        "-polarization",
        "NO",
    ])
'''

# check
ell = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat")[:1501,0]
#clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat")[:1501,1]
clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz_fullmission.dat")[:1501,1]
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clkk_pr4.dat")[:1501,1]
clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clik_545ghz_pr4.dat")[:1501,1]
#clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/separate_masks/cls_clii_545ghz.dat")[:1501,1]
#clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/separate_masks/cls_clkk_pr4.dat")[:1501,1]
#clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/separate_masks/cls_clik_545ghz_pr4.dat")[:1501,1]
#clii_pr3 = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/separate_masks/cls_clii_545ghz_fionamask.dat")[:1501,1] * (1e6/58.04)**2 # MJy/sr -> uK_CMB
#clkk_pr3 = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/separate_masks/cls_clkk_pr3.dat")[:1501,1]
#clik_pr3 = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/separate_masks/cls_clik_545ghz_pr3_fionamask.dat")[:1501,1] * 1e6/58.04
# cib spectra from BKSPT paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0) * (1e6/58.04) # MJy/sr -> uK_CMB
clii_bkspt = np.nanmean(cib['cibauto'], axis=0) * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk_bkspt = cib['clkk']

## try simple anafast
#cib_fiona_map_rh1 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH1.fits")
#cib_fiona_map_rh2 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_RH2.fits")
#kap_pr4_PP = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4PP_nside2048.fits")
#kap_pr4_TT = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits")
#cib_fiona_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/CIB_3tracers_545_mask1_nside16NILC_full.fits")
#kap_pr4 = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits")
#cib_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask.fits")
#kap_mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits")
##mask_common = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/545mask_pr4kappamask_common_v2.fits")
##fsky = np.mean(mask_common**2)
#cib_fiona_map_rh1 *= cib_mask
#cib_fiona_map_rh2 *= cib_mask
#cib_fiona_map *= cib_mask
#kap_pr4_PP *= kap_mask
#kap_pr4_TT *= kap_mask
#kap_pr4 *= kap_mask
#cib_fiona_map_rh1 = np.nan_to_num(cib_fiona_map_rh1, nan=0.0)
#cib_fiona_map_rh2 = np.nan_to_num(cib_fiona_map_rh2, nan=0.0)
#cib_fiona_map = np.nan_to_num(cib_fiona_map, nan=0.0)
#clii_anafast = hp.anafast(cib_fiona_map_rh1,cib_fiona_map_rh2) / (np.mean(cib_mask**2))
#clkk_anafast = hp.anafast(kap_pr4_PP,kap_pr4_TT) / (np.mean(kap_mask**2))
#clik_anafast = hp.anafast(kap_pr4,cib_fiona_map) / (np.mean(cib_mask*kap_mask))

# smooth
fwhm_dell = 40 # smoothing width in multipoles (FWHM)
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma
clik_bkspt = gaussian_filter1d(clik_bkspt, sigma=sigma, mode='nearest')
clii_bkspt = gaussian_filter1d(clii_bkspt, sigma=sigma, mode='nearest')
clkk_bkspt = gaussian_filter1d(clkk_bkspt, sigma=sigma, mode='nearest')
clik = gaussian_filter1d(clik, sigma=sigma, mode='nearest')
clii = gaussian_filter1d(clii, sigma=sigma, mode='nearest')
clkk = gaussian_filter1d(clkk, sigma=sigma, mode='nearest')
#clik_anafast = gaussian_filter1d(clik_anafast, sigma=sigma, mode='nearest')
#clii_anafast = gaussian_filter1d(clii_anafast, sigma=sigma, mode='nearest')
#clkk_anafast = gaussian_filter1d(clkk_anafast, sigma=sigma, mode='nearest')

# rho
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,1500)
clkk_theory = slpp * (np.arange(1501)*(np.arange(1501)+1))**2/4
#rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
#rho = clik / np.sqrt(clii * clkk)
rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
rho = clik / np.sqrt(clii * clkk_theory)
#rho_anafast = clik_anafast / np.sqrt(clii_anafast * clkk_anafast)
print('rho_bkspt: ', rho_bkspt)
print('rho: ', rho)
#print('rho_anafast: ', rho_anafast)

# plot
plt.figure(0)
plt.clf()

#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(np.arange(1501), clik_bkspt/clik[:1501], color='lightcoral', alpha=0.5, label='clik BKSPT / clik')
#plt.plot(np.arange(1501), clkk_bkspt/clkk[:1501], color='cornflowerblue', alpha=0.5, label='clkk BKSPT / clkk')
#plt.plot(np.arange(1501), clii_bkspt/clii[:1501], color='orange', alpha=0.5, label='clii BKSPT / clii')
#plt.ylim(0,100)
#plt.yscale('log')

#plt.plot(np.arange(1501), clik_bkspt, color='firebrick', alpha=0.5, label='clik BKSPT')
#plt.plot(np.arange(1501), clkk_bkspt, color='darkblue', alpha=0.5, label='clkk BKSPT')
#plt.plot(np.arange(1501), clii_bkspt, color='darkorange', alpha=0.5, label='clii BKSPT')
#plt.plot(np.arange(1501), clik[:1501], color='lightcoral', alpha=0.5, linestyle='--', label='clik')
#plt.plot(np.arange(1501), clkk[:1501], color='cornflowerblue', alpha=0.5, linestyle='--', label='clkk')
#plt.plot(np.arange(1501), clii[:1501], color='orange', alpha=0.5, linestyle='--', label='clii')
##plt.plot(np.arange(1501), clii_alm2cl[:1501], color='plum', alpha=0.5, linestyle='--', label='clii from alm2cl')
#plt.yscale('log')

plt.plot(np.arange(1501), rho_bkspt[:1501], color='firebrick', alpha=0.5, label='rho BKSPT')
plt.plot(np.arange(1501), rho[:1501], color='lightcoral', alpha=0.5, label='rho')
#plt.plot(np.arange(1501), rho_anafast[:1501], color='thistle', linestyle='--', alpha=1, label='rho from anafast')
plt.ylim(0,2)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,1500)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/fiona_545ghz_cib_pr4_kappa_specs.png')
