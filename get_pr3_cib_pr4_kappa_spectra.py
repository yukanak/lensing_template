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
#hdu = fits.open("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask.fits.gz")
#m = hdu[1].data.field(0).astype(float).ravel()
#hp.write_map("/home/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits", m, overwrite=True)
#klm = hp.read_alm("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/PR42018like_klm_dat_MV.fits")
#kappa_map = hp.alm2map(klm, nside=nside)
#hp.write_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits", kappa_map)


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

    clfile = f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_pr4kappa_patch{i}.dat"
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
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits",
            "-maskfile2",
            "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
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
    #subprocess.call(
    #    [
    #        spice,
    #        "-mapfile",
    #        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4MV_nside2048.fits",
    #        "-maskfile",
    #        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
    #        "-pixelfile",
    #        "YES",
    #        "-mapfile2",
    #        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits",
    #        "-maskfile2",
    #        cib_mask,
    #        "-beam2",
    #        "5",
    #        "-pixelfile2",
    #        "YES",
    #        "-clfile",
    #        clfile,
    #        "-nlmax",
    #        "2048",
    #        "-apodizesigma",
    #        "15",
    #        "-thetamax",
    #        "20",
    #        "-subdipole",
    #        "YES",
    #        "-subav",
    #        "YES",
    #        "-apodizetype",
    #        "0",
    #        "-verbosity",
    #        "NO",
    #        "-polarization",
    #        "NO",
    #    ])

# clkk
subprocess.call(
    [
        spice,
        "-mapfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4PP_nside2048.fits",
        "-maskfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "-pixelfile",
        "YES",
        "-mapfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/kappa_map_PR4TT_nside2048.fits",
        "-maskfile2",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/PR42018like_maps/PR4_variations/mask_scalar.fits",
        "-pixelfile2",
        "YES",
        "-clfile",
        "/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr4/cls/cls_clkk_pr4.dat",
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

#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(np.arange(2,1501), clik_bkspt/clik_pr3[:1499], color='firebrick', alpha=0.5, label='clik BKSPT / clik WITH PR3 KAPPA')
#plt.plot(np.arange(2,1501), clik_bkspt/clik[:1499], color='lightcoral', alpha=0.5, label='clik BKSPT / clik')
#plt.plot(np.arange(2,1501), clkk_bkspt/clkk[:1499], color='cornflowerblue', alpha=0.5, label='clkk BKSPT / clkk')
#plt.plot(np.arange(2,1501), clii_bkspt/clii[:1499], color='orange', alpha=0.5, label='clii BKSPT / clii')
#plt.ylim(0.8,1.2)

plt.plot(np.arange(2,1501), clik_bkspt, color='firebrick', alpha=0.5, label='clik BKSPT loaded')
#plt.plot(np.arange(2,1501), clkk_bkspt, color='darkblue', alpha=0.5, label='clkk BKSPT loaded')
plt.plot(np.arange(2,1501), clii_bkspt, color='darkorange', alpha=0.5, label='clii BKSPT loaded')
plt.plot(np.arange(2,lmax+1), clik[:lmax+1], color='lightcoral', alpha=0.5, linestyle='--', label='new clik with PR4 kappa')
#plt.plot(np.arange(2,lmax+1), clkk[:lmax+1], color='cornflowerblue', alpha=0.5, linestyle='--', label='clkk')
plt.plot(np.arange(2,lmax+1), clii[:lmax+1], color='orange', alpha=0.5, linestyle='--', label='new clii (still PR3 GNILC CIB)')
plt.yscale('log')

#plt.plot(np.arange(1501), rho_bkspt[:1501], color='firebrick', alpha=0.5, label='rho from curves loaded from BKSPT delensing paper analysis')
#plt.plot(np.arange(lmax+1), rho[:lmax+1], color='lightcoral', alpha=0.5, label='new rho with PR4 kappa')
#plt.ylim(0,1.25)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='medium')
plt.xscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/pr3_545ghz_cib_pr4_kappa_specs.png')
