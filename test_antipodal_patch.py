import sys, os
import numpy as np
import healpy as hp
sys.path.insert(0, "/home/users/yukanaka/miniconda3/envs/tora_py3/lib/python3.10/site-packages/")
import h5py
from astropy.io import fits
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_array
import yaml
sys.path.insert(0,'/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as hutils
import qest
import matplotlib.pyplot as plt

def tp2rd(tht, phi):
    ra=phi/np.pi*180.0
    dec=((tht*-1)+np.pi/2.0)/np.pi*180.
    return ra,dec

def rd2tp(ra,dec):
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

idx = 1
nside = 2048
cmbdir="/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/lmax5000/"
fname="lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed%i_lmax17000_nside8192_interp1.6_method1_pol_1_alms_lowpass5000.fits"%idx
filemask = '/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits'
blm = hp.read_alm(cmbdir+fname, hdu=[3])
mask = hp.read_map(filemask)
pix = np.where(mask==1.0)[0]
pixs={}
pixs[1] = pix
tht,phi = hp.pix2ang(nside,pix)

# Map-space cut (same blm, cut on antipodal pixels)
bmap = hp.alm2map(blm, nside, lmax=5000)
ra,dec = tp2rd(tht,phi+np.pi)
tht4,phi4 = rd2tp(ra,-1*dec)
pixs[2] = hp.ang2pix(nside,tht4,phi4)
b_patch2 = bmap[pixs[2]]

# Alm-space antipode (antipode-transformed blm, cut on original pixels)
ell, m = hp.Alm.getlm(5000)
phase = (-1.0)**ell
blm_anti = hp.almxfl(blm, -phase)
bmap_anti = hp.alm2map(blm_anti, nside=nside, lmax=5000)
b_patch2_alt = bmap_anti[pixs[1]]

diff = b_patch2 - b_patch2_alt
print("RMS difference:", np.std(diff))
print("Max abs difference:", np.max(np.abs(diff)))

