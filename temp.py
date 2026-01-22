import subprocess, os, sys
import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from astropy import constants as const
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

nside = 2048
os.environ["HEALPIX"] = "/home/users/yukanaka/miniconda3/envs/tora_py3/Healpix_3.83/"
spice = "/home/users/yukanaka/PolSpice_v03-08-03/bin/spice"
#kmap = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/kappa_map_PR3_nside2048.fits")
#mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/mask_scalar.fits")
#kmap_bkspt = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/dat_k_map.fits")
#klm = hp.read_alm("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_Lensing_2048_R2.00/data/dat_klm.fits")

for i in [0,1,2,3,5,6,7,8]: # 8 patches
    cib_mask = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/190717_cibtest/mask_patch_{i}.fits"

    clfile = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/cls_by_yuka/cls_clik_545ghz_patch{i}.dat"
    # clik
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
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/dat_k_map.fits",
            "-maskfile2",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/mask.fits",
            "-pixelfile2",
            "YES",
            "-clfile",
            clfile,
            "-nlmax",
            "1500",
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

# check
lmax = 2000
ell = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/cls/cls_clkk_pr4.dat")[:lmax+1,0]
cliis = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clii_545ghz_patch{i}.dat")[:lmax+1,1] * (1e6/58.04)**2 for i in [0,1,2,3,5,6,7,8]])
clii = np.nanmean(cliis, axis=0)
cliks = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_pr4kappa_patch{i}.dat")[:lmax+1,1] * (1e6/58.04) for i in [0,1,2,3,5,6,7,8]])
clik = np.nanmean(cliks, axis=0)
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/cls/cls_clkk_pr4.dat")[:lmax+1,1]
# cib spectra from BKSPT paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
clik_bkspt = np.nanmean(cib['cibxkap'], axis=0) * (1e6/58.04) # MJy/sr -> uK_CMB
clii_bkspt = np.nanmean(cib['cibauto'], axis=0) * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk_bkspt = cib['clkk']
# clik with PR3 CIB + PR3 kappa
cliks_pr3 = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_patch{i}.dat")[:1501,1] * (1e6/58.04) for i in [0,1,2,3,5,6,7,8]])
clik_pr3 = np.nanmean(cliks_pr3, axis=0)
# clik with BKSPT "dat_k_map.fits"
cliks_bkspt_yuka = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/bksptpol_delens/cls_by_yuka/cls_clik_545ghz_patch{i}.dat")[:lmax+1,1] * (1e6/58.04) for i in [0,1,2,3,5,6,7,8]])
clik_bkspt_yuka = np.nanmean(cliks_bkspt_yuka, axis=0)
# theory clkk
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk_theory = slpp * (np.arange(lmax+1)*(np.arange(lmax+1)+1))**2/4

# smooth
def bin_interp_spectra(spectrum, li=np.arange(2,2001), average_window=81):
    '''
    from cib_utils.py
    spectrum: unbinned spectrum
    li      : range of ell to be interpolated
    '''
    # Moving average to suppress noise and jaggedness
    spectrum_filtered = savgol_filter(spectrum, average_window, 0)
    # Interpolation to guarantee continuity and smooth derivatives, and to allow consistent evaluation for almxfl and synalm
    cl_spline_mean = InterpolatedUnivariateSpline(np.arange(0, len(spectrum_filtered)), spectrum_filtered)
    cl_li = cl_spline_mean(li)
    return cl_li
li = np.arange(2,lmax+1)
clkk_spline = InterpolatedUnivariateSpline(ell, clkk_theory)
clkk_theory = clkk_spline(li)
clii = bin_interp_spectra(clii, li)
clik = bin_interp_spectra(clik, li)
clkk = bin_interp_spectra(clkk, li)
clik_pr3 = bin_interp_spectra(clik_pr3, li)
clik_bkspt_yuka = bin_interp_spectra(clik_bkspt_yuka, li)
li = np.arange(2,1500+1)
clii_bkspt = bin_interp_spectra(clii_bkspt, li)
clik_bkspt = bin_interp_spectra(clik_bkspt, li)
clkk_bkspt = bin_interp_spectra(clkk_bkspt, li)

# rho
li = np.arange(2,1500+1)
rho_bkspt_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(clik_bkspt / np.sqrt(clii_bkspt * clkk_bkspt)))
rho_bkspt = rho_bkspt_spline(ell)
li = np.arange(2,lmax+1)
rho_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(clik / np.sqrt(clii * clkk_theory)))
rho = rho_spline(ell)
rho_pr3_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(clik_pr3 / np.sqrt(clii * clkk_theory)))
rho_pr3 = rho_pr3_spline(ell)
print('rho_bkspt: ', rho_bkspt)
print('rho: ', rho)

# plot
plt.figure(0)
plt.clf()

plt.axhline(y=1, color='gray', linestyle='--')
plt.plot(np.arange(2,1501), clik_bkspt/clik_pr3[:1499], color='firebrick', alpha=0.5, label='clik BKSPT / clik WITH PR3 KAPPA')
plt.plot(np.arange(2,1501), clik_bkspt/clik[:1499], color='lightcoral', alpha=0.5, label='clik BKSPT / clik')
plt.plot(np.arange(2,1501), clik_bkspt/clik_bkspt_yuka[:1499], color='slateblue', alpha=0.5, label='clik BKSPT / clik WITH KAPPA FROM KIMMY')
#plt.plot(np.arange(2,1501), clkk_bkspt/clkk[:1499], color='cornflowerblue', alpha=0.5, label='clkk BKSPT / clkk')
#plt.plot(np.arange(2,1501), clii_bkspt/clii[:1499], color='orange', alpha=0.5, label='clii BKSPT / clii')
plt.ylim(0.5,1.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='small')
plt.xscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')


