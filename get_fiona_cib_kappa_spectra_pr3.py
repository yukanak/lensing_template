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
#klm = hp.read_alm("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/dat_klm.fits")
#kappa_map = hp.alm2map(klm, nside=nside)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits", kappa_map)
#hdu = fits.open("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask.fits.gz")
#m = hdu[1].data.field(0).astype(float).ravel()
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits", m, overwrite=True)

'''
for i in [0,1,2,3,5,6,7,8]: # 8 patches
    cib_mask = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/190717_cibtest/mask_patch_{i}.fits"

    clfile = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clii_545ghz_patch{i}.dat"
    # clii
    subprocess.call(
        [
            spice,
            "-mapfile",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits",
            "-maskfile",
            cib_mask,
            "-beam",
            "5",
            "-pixelfile",
            "YES",
            "-mapfile2",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits",
            "-maskfile2",
            cib_mask,
            "-beam2",
            "5",
            "-pixelfile2",
            "YES",
            "-clfile",
            clfile,
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

    clfile = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_patch{i}.dat"
    # clik
    subprocess.call(
        [
            spice,
            "-mapfile",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits",
            "-maskfile",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits",
            "-pixelfile",
            "YES",
            "-mapfile2",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits",
            "-maskfile2",
            cib_mask,
            "-beam2",
            "5",
            "-pixelfile2",
            "YES",
            "-clfile",
            clfile,
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
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits",
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits",
        "-pixelfile",
        "YES",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits",
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits",
        "-pixelfile2",
        "YES",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clkk_pr3.dat",
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
clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clii_545ghz.dat")[:1501,1]
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clkk_pr4.dat")[:1501,1]
clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/fiona_cib/cls/cls_clik_545ghz_pr4.dat")[:1501,1]
cliis_pr3 = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clii_545ghz_patch{i}.dat")[:1501,1] * (1e6/58.04)**2 for i in [0,1,2,3,5,6,7,8]])
clii_pr3 = np.nanmean(cliis_pr3, axis=0)
cliks_pr3 = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_patch{i}.dat")[:1501,1] * (1e6/58.04) for i in [0,1,2,3,5,6,7,8]])
clik_pr3 = np.nanmean(cliks_pr3, axis=0)
clkk_pr3 = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clkk_pr3.dat")[:1501,1]
# cib spectra from BKSPT paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0) * (1e6/58.04) # MJy/sr -> uK_CMB
clii_bkspt = np.nanmean(cib['cibauto'], axis=0) * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk_bkspt = cib['clkk']

# smooth
fwhm_dell = 40 # smoothing width in multipoles (FWHM)
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma
clik_bkspt = gaussian_filter1d(clik_bkspt, sigma=sigma, mode='nearest')
clii_bkspt = gaussian_filter1d(clii_bkspt, sigma=sigma, mode='nearest')
clkk_bkspt = gaussian_filter1d(clkk_bkspt, sigma=sigma, mode='nearest')
clik = gaussian_filter1d(clik, sigma=sigma, mode='nearest')
clii = gaussian_filter1d(clii, sigma=sigma, mode='nearest')
clkk = gaussian_filter1d(clkk, sigma=sigma, mode='nearest')
clik_pr3 = gaussian_filter1d(clik_pr3, sigma=sigma, mode='nearest')
clii_pr3 = gaussian_filter1d(clii_pr3, sigma=sigma, mode='nearest')
clkk_pr3 = gaussian_filter1d(clkk_pr3, sigma=sigma, mode='nearest')

# rho
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,1500)
clkk_theory = slpp * (np.arange(1501)*(np.arange(1501)+1))**2/4
#rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
#rho = clik / np.sqrt(clii * clkk)
#rho_pr3 = clik_pr3 / np.sqrt(clii_pr3 * clkk_pr3)
rho_bkspt = clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)
rho = clik / np.sqrt(clii * clkk_theory)
rho_pr3 = clik_pr3 / np.sqrt(clii_pr3 * clkk_theory)
print('rho_bkspt: ', rho_bkspt)
print('rho: ', rho)
print('rho_pr3: ', rho_pr3)

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
#plt.plot(np.arange(1501), clik_pr3[:1501], color='thistle', alpha=1, linestyle='--', label='clik PR3')
#plt.plot(np.arange(1501), clkk_pr3[:1501], color='powderblue', alpha=1, linestyle='--', label='clkk PR3')
#plt.plot(np.arange(1501), clii_pr3[:1501], color='bisque', alpha=1, linestyle='--', label='clii PR3')
#plt.yscale('log')

plt.plot(np.arange(1501), rho_bkspt[:1501], color='firebrick', alpha=0.5, label='rho BKSPT')
plt.plot(np.arange(1501), rho[:1501], color='lightcoral', alpha=0.5, label='rho')
plt.plot(np.arange(1501), rho_pr3[:1501], color='pink', linestyle='--', alpha=0.5, label='rho from pr3')
plt.ylim(0,2)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,1500)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/fiona_545ghz_cib_pr4_kappa_specs.png')
