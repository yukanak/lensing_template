import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/users/yukanaka/spt3g_software/scratch/wlwu/cinv_lowell/')
sys.path.insert(0, '/home/users/yukanaka/healqest/healqest/src/')
sys.path.insert(0, '/home/users/yukanaka/healqest/pipeline/')
import maps as hq_maps
import healqest_utils as utils
from cinv import cinv_hp as cinv
from cinv import cinv_hp_yuka as cinv_old
import cinv_lowell as cll
from spt3g.lensing import hp_utils

# See /home/users/yukanaka/spt3g_software/scratch/wlwu/cinv_lowell/script24_0830_ivf_invvar.py

params = yaml.safe_load(open("/home/users/yukanaka/lensing_template/lowell_v3mocks_musebeamv41.yaml"))
params_corr = yaml.safe_load(open("/home/users/yukanaka/lensing_template/lowell_v3mocks_musebeamv41_corr.yaml"))

outdir = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/'
outdir_corr = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_corr/cinv_output_test_corr/'

nlev_p_uk = 5/np.sqrt(2) # at ell < 500, more like 10 uK-arcmin
nvar      = (nlev_p_uk*np.pi/180./60.)**2 # ~ 1e-6 uK^2-steradian
lmax      = params['lmax']
lmin      = 20
nside     = 2048
mask  = cll.get_mask(params)
fsky  = cll.get_fsky()
l = np.arange(lmax + 1)

nl2d_ee = cll.get_smooth_nl2d(params, eorb='ee')#/fsky
nl2d_bb = cll.get_smooth_nl2d(params, eorb='bb')#/fsky
nl2d_ee_corr = cll.get_smooth_nl2d(params_corr, eorb='ee')#/fsky
nl2d_bb_corr = cll.get_smooth_nl2d(params_corr, eorb='bb')#/fsky

nl2d_ee_alt = cll.get_inv_nvar_weighted_nl2d(params, eorb='ee')
nl2d_bb_alt = cll.get_inv_nvar_weighted_nl2d(params, eorb='bb')

nl2d_ee_grid = hp_utils.alm2grid(nl2d_ee,lmax)
nl2d_bb_grid = hp_utils.alm2grid(nl2d_bb,lmax)
nl2d_ee_grid_corr = hp_utils.alm2grid(nl2d_ee_corr,lmax)
nl2d_bb_grid_corr = hp_utils.alm2grid(nl2d_bb_corr,lmax)
nl2d_ee_alt_grid = hp_utils.alm2grid(nl2d_ee_alt,lmax)
nl2d_bb_alt_grid = hp_utils.alm2grid(nl2d_bb_alt,lmax)

# error bar estimate with n_ell only, no c_ell (which should cancel)
ell, emm = hp.Alm.getlm(lmax)
nl1d = np.bincount(ell, weights=nl2d_ee, minlength=lmax+1) / (2*l+1)
nl1d_corr = np.bincount(ell, weights=nl2d_ee_corr, minlength=lmax+1) / (2*l+1)
errorbar = nl1d * np.sqrt(2/(2*l+1)/fsky)
errorbar_corr = nl1d_corr * np.sqrt(2/(2*l+1)/fsky)
#np.save('errorbar.npy', errorbar)
#np.save('errorbar_corr.npy', errorbar_corr)

# plot check
plt.clf()
plt.figure()
plt.imshow(np.log10(nl2d_ee_grid.T), origin='lower', aspect='auto',extent=[0, lmax, 0, lmax],cmap='viridis')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$m$')
plt.title('log10 of nl2d_ee, get_smooth_nl2d')
plt.colorbar(label='amplitude')
plt.tight_layout()
plt.savefig('figs/nl2d_ee.png')

plt.clf()
plt.figure()
plt.imshow(np.log10(nl2d_bb_grid.T), origin='lower', aspect='auto',extent=[0, lmax, 0, lmax],cmap='viridis')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$m$')
plt.title('log10 of nl2d_bb, get_smooth_nl2d')
plt.colorbar(label='amplitude')
plt.tight_layout()
plt.savefig('figs/nl2d_bb.png')

plt.clf()
plt.figure()
plt.imshow(np.log10(nl2d_ee_alt_grid.T), origin='lower', aspect='auto',extent=[0, lmax, 0, lmax],cmap='viridis')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$m$')
plt.title('log10 of nl2d_ee, get_inv_nvar_weighted_nl2d')
plt.colorbar(label='amplitude')
plt.tight_layout()
plt.savefig('figs/nl2d_ee_alt.png')

plt.clf()
plt.figure()
plt.imshow(np.log10(nl2d_bb_alt_grid.T), origin='lower', aspect='auto',extent=[0, lmax, 0, lmax],cmap='viridis')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$m$')
plt.title('log10 of nl2d_bb, get_inv_nvar_weighted_nl2d')
plt.colorbar(label='amplitude')
plt.tight_layout()
plt.savefig('figs/nl2d_bb_alt.png')

# slices
plt.clf()
ell_values = [50, 200, 500]
for ell in ell_values:
    m_vals = np.arange(ell + 1)
    idxs = hp.Alm.getidx(lmax, ell, m_vals)
    nl2d_ee_slice = nl2d_ee[idxs]
    nl2d_ee_corr_slice = nl2d_ee_corr[idxs]
    nl2d_ee_alt_slice = nl2d_ee_alt[idxs]
    plt.plot(m_vals, nl2d_ee_slice, label=rf'$\ell={ell}$, uncorr')#get_smooth_nl2d')
    #plt.plot(m_vals, nl2d_ee_alt_slice, linestyle='--', label=rf'$\ell={ell}$, get_inv_nvar_weighted_nl2d')
    plt.plot(m_vals, nl2d_ee_corr_slice, linestyle='--', label=rf'$\ell={ell}$, corr')
plt.xlabel(r'$m$')
plt.title('nl2d_ee')
plt.legend()
plt.title('slices at fixed $\ell$')
plt.tight_layout()
plt.savefig('figs/nl2d_ee_slices.png')

plt.clf()
ell_values = [50, 200, 500]
for ell in ell_values:
    m_vals = np.arange(ell + 1)
    idxs = hp.Alm.getidx(lmax, ell, m_vals)
    nl2d_bb_slice = nl2d_bb[idxs]
    nl2d_bb_alt_slice = nl2d_bb_alt[idxs]
    plt.plot(m_vals, nl2d_bb_slice, label=rf'$\ell={ell}$, get_smooth_nl2d')
    plt.plot(m_vals, nl2d_bb_alt_slice, linestyle='--', label=rf'$\ell={ell}$, get_inv_nvar_weighted_nl2d')
plt.xlabel(r'$m$')
plt.title('nl2d_bb')
plt.legend()
plt.title('slices at fixed $\ell$')
plt.tight_layout()
plt.savefig('figs/nl2d_bb_slices.png')
