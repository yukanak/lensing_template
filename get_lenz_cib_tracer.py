#!/usr/bin/env python
import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('idx'       , default=None, type=int, help='idx')
args = parser.parse_args()
idx = args.idx
print(idx)

def tp2rd(tht, phi):
    ra=phi/np.pi*180.0
    dec=((tht*-1)+np.pi/2.0)/np.pi*180.
    return ra,dec

def rd2tp(ra, dec):
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

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

#lmax = 1500
lmax = 2000
nside = 2048
l = np.arange(lmax+1)
dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_tracers/lenz_cib_phi_tracer/'
clii = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clii.dat")[:lmax+1,1]# * (1e6/58.04)**2 # MJy/sr -> uK_CMB
clkk = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clkk_pr4.dat")[:lmax+1,1]
clik = np.loadtxt("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/cls/cls_clik_pr4.dat")[:lmax+1,1]# * (1e6/58.04) # MJy/sr -> uK_CMB
# Theory clkk
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk_theory = slpp * (np.arange(lmax+1)*(np.arange(lmax+1)+1))**2/4
# Smoothing
li = np.arange(2,2001)
clkk_spline = InterpolatedUnivariateSpline(ell, clkk_theory)
clkk_theory = clkk_spline(li)
clii = bin_interp_spectra(clii, li)
clik = bin_interp_spectra(clik, li)
clkk = bin_interp_spectra(clkk, li)

# MAKE CIB SIMS FROM INPUT PHI (https://arxiv.org/pdf/2011.08163)
if idx <= 250:
    idx_new = idx
else:
    idx_new = idx - 250
# Load plm and convert to klm
input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_4096/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{idx_new}_lmax4096.alm')
input_plm = utils.reduce_lmax(input_plm, lmax=lmax)
ell, m = hp.Alm.getlm(lmax)
fac = np.zeros_like(ell, dtype=float)
fac[ell>=2] = 0.5 * ell[ell>=2]*(ell[ell>=2]+1.0)
input_klm = hp.almxfl(input_plm, fac)
input_kmap = hp.alm2map(input_klm, nside=nside, lmax=lmax)
# noise
nl_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(clii - clik**2/clkk_theory))
nl = nl_spline(l)
#nl = clii - clik**2/clkk_theory
#nl = np.clip(nl, 0, None)
I_noi = hp.synalm(nl, lmax=lmax, new=True)
I_noi_map = hp.alm2map(I_noi, nside=nside, lmax=lmax)
# signal
ik_over_kk_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(clik/clkk_theory))
ik_over_kk = ik_over_kk_spline(l)
I_sig = hp.almxfl(input_klm, ik_over_kk)
#I_sig = hp.almxfl(input_klm, clik/clkk_theory)
I_sig_map = hp.alm2map(I_sig, nside=nside, lmax=lmax)
# IF idx > 250, need ANTIPODE of input plm
if idx > 250:
    # https://github.com/SouthPoleTelescope/spt3g_software/blob/511f58f03a0a3e53f06f0ebd5d2df31bd6a33743/scratch/yomori/utils/utils.py#L248
    mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits')
    pix = np.where(mask > 0)[0] # patch 1 pixel list
    tht,phi = hp.pix2ang(nside,pix)
    tht2,phi2 = tht,phi+np.pi
    ra,dec = tp2rd(tht2,phi2) # rotate 180
    tht4,phi4 = rd2tp(ra,-1*dec) # flip
    pix_antipode = hp.ang2pix(nside,tht4,phi4)
    I_sig_map_new = np.zeros_like(I_sig_map)
    I_sig_map_new[pix] = I_sig_map[pix_antipode] # full-sky map whose values over patch 1 are the values from patch 2
    I_sig_map = I_sig_map_new
# total
#I_alm = I_sig + I_noi
#I_map = hp.alm2map(I_alm, nside=nside, lmax=lmax)
I_map = I_sig_map + I_noi_map
# SAVE CIB-BASED PHI TRACER AS JUST THE I, NO WIENER FILTERING HERE
hp.write_map(dir_out+f'cib_tracer_seed{idx}.fits', I_map)#, overwrite=True)

'''
# CHECK
clii_sim1 = hp.anafast(I_map,lmax=lmax)
clik_sim1 = hp.anafast(I_map,input_kmap,lmax=lmax)
clkk_sim1 = hp.alm2cl(input_klm)
#clkk_sim1 = hp.anafast(input_kmap, lmax=lmax)
rho_sim1 = clik_sim1 / np.sqrt(clii_sim1 * clkk_sim1)
# Plot
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, clik_sim1/clik, color='lightcoral', alpha=0.5, label='clik sim 1 / clik')
plt.plot(l, clkk_sim1/clkk, color='cornflowerblue', alpha=0.5, label='clkk sim 1 / clkk')
plt.plot(l, clii_sim1/clii, color='orange', alpha=0.5, label='clii sim 1 / clii')
plt.plot(l, rho_sim1/(clik/np.sqrt(clii*clkk)), linestyle='--', alpha=0.5, color='lightgreen', label='rho ratio sim 1 / (clik / sqrt(clii*clkk))')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.ylim(-0.01, 2)
plt.xscale('log')
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/cib_sims.png')
'''
