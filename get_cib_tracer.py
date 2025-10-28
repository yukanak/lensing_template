#!/usr/bin/env python
import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
from scipy.ndimage import gaussian_filter1d
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

lmax = 1500
nside = 2048
l = np.arange(lmax+1)
dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_phi_pp_tracer/'
# there are 8 cib['cibxkap'] and cib['cibauto'] spectra in the file
# from the 8 high Galactic latitude patches as noted in the BKSPT delensing paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
#rho = cib['cibxkap']/np.sqrt(cib['cibauto']*cib['clkk'])
# to start, you can take the mean of them as the cib x kappa and cib-auto values
clik = np.nanmean(cib['cibxkap'], axis=0)
clii = np.nanmean(cib['cibauto'], axis=0)
clkk = cib['clkk']
# the spectra have a lot of wiggles, which are not real structure
# (patches are small so the minimum resolution in ell is 10s of ells)
# so smoothing before feeding them to the rest of the pipeline is good
# Gaussian smoothing with kernel that corresponds to \Delta_ell = 30 to 50 would be reasonable
fwhm_dell = 40 # smoothing width in multipoles (FWHM)
sigma = fwhm_dell / (2*np.sqrt(2*np.log(2))) # convert FWHM to sigma
clik = gaussian_filter1d(clik, sigma=sigma, mode='nearest')
clii = gaussian_filter1d(clii, sigma=sigma, mode='nearest')
clkk = gaussian_filter1d(clkk, sigma=sigma, mode='nearest')

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
nl = clii - clik**2/clkk
nl = np.clip(nl, 0, None)
I_noi = hp.synalm(nl, lmax=lmax, new=True)
I_noi_map = hp.alm2map(I_noi, nside=nside, lmax=lmax)
# signal
I_sig = hp.almxfl(input_klm, clik/clkk)
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
plt.xscale('log')
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/cib_sims.png')
'''

'''
# MAKE CIB-BASED PHI TRACER
#wiener_filter = np.zeros_like(clik); wiener_filter[2:] = clik[2:]/clii[2:]
#klm = hp.almxfl(I_alm, wiener_filter)
#hp.write_alm(dir_out+f'cib_tracer_alm_seed{idx}.alm', klm)

# CHECK
clik_sim1 = hp.alm2cl(klm,I_alm) # should match clik
clkk_sim1 = hp.alm2cl(klm) # should match clik**2/clii
clkk_input_sim1 = hp.alm2cl(klm,input_klm) # should match clik**2/clii
rho_sim1 = clkk_input_sim1 / np.sqrt(clkk_sim1 * hp.alm2cl(input_klm))
# Plot
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, clik_sim1/clik, color='lightcoral', alpha=0.5, label='clik sim 1 / clik')
plt.plot(l, clkk_sim1/(clik**2/clii), color='cornflowerblue', alpha=0.5, label='clkk sim 1 / (wiener filter**2 * clii)')
plt.plot(l, clkk_input_sim1/(clik**2/clii), color='orange', alpha=0.5, label='clkk sim 1 x input klm / (wiener filter**2 * clii)')
plt.plot(l, rho_sim1/(clik/np.sqrt(clii*clkk)), linestyle='--', alpha=0.5, color='lightgreen', label='rho ratio: (clkk_input_sim1 / sqrt(clkk_sim1*clkk_input)) / (clik / sqrt(clii*clkk))')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/cib_sims.png')
'''

