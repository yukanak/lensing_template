#!/usr/bin/env python
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
import argparse
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

# https://arxiv.org/pdf/2212.07420 equations 89 - 92
# https://arxiv.org/pdf/1705.02332 equations 7 - 9
yaml_file = 'bt_gmv3500_combined_pp.yaml'
btmp = bt.btemplate(yaml_file)
#lmax = btmp.lmax_b
lmax = 1500
nside = btmp.nside
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size
l = np.arange(lmax+1)

# Get reconstructed 2019/2020 analysis phi tracer
klm1 = btmp.get_debiased_klm(idx)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)
klm1_map = hp.alm2map(klm1, nside)
auto1 = hp.anafast(klm1_map * mask, lmax=lmax)/fsky

# Get CIB-based phi tracer
#klm2 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_phi_tracer/cib_klm_seed{idx}.alm')
#klm2_map = hp.alm2map(klm2, nside)
klm2_map = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_phi_pp_tracer/cib_tracer_seed{idx}.fits')
#TODO: there will be ringing here...
klm2 = hp.map2alm(klm2_map, lmax=lmax)
auto2 = hp.anafast(klm2_map * mask, lmax=lmax)/fsky
cross12 = hp.anafast(klm1_map * mask, klm2_map * mask, lmax=lmax)/fsky

# Get input kappa
# Use fiducial for kappa auto for less noise
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
autok = slpp * (l*(l+1))**2/4 # TODO: OR autok = hp.anafast(input_kmap * mask, lmax=lmax)/fsky OR hp.alm2cl(input_klm)
# Load per-realization input plm to cross with tracers
if idx > 250:
    idx_new = idx - 250
else:
    idx_new = idx
input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_4096/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{idx_new}_lmax4096.alm')
input_plm = utils.reduce_lmax(input_plm, lmax=lmax)
ell, m = hp.Alm.getlm(lmax)
fac = np.zeros_like(ell, dtype=float)
fac[ell>=2] = 0.5 * ell[ell>=2]*(ell[ell>=2]+1.0)
input_klm = hp.almxfl(input_plm, fac)
input_kmap = hp.alm2map(input_klm, nside, lmax=lmax)
# IF idx > 250, need ANTIPODE of input plm
if idx > 250:
    # https://github.com/SouthPoleTelescope/spt3g_software/blob/511f58f03a0a3e53f06f0ebd5d2df31bd6a33743/scratch/yomori/utils/utils.py#L248
    pix = np.where(mask > 0)[0] # patch 1 pixel list
    tht,phi = hp.pix2ang(nside,pix)
    tht2,phi2 = tht,phi+np.pi
    ra,dec = tp2rd(tht2,phi2) # rotate 180
    tht4,phi4 = rd2tp(ra,-1*dec) # flip
    pix_antipode = hp.ang2pix(nside,tht4,phi4)
    input_kmap_new = np.zeros_like(input_kmap)
    input_kmap_new[pix] = input_kmap[pix_antipode] # full-sky map whose values over patch 1 are the values from patch 2
    input_kmap = input_kmap_new
cross1k = hp.anafast(klm1_map * mask, input_kmap * mask, lmax=lmax)/fsky
cross2k = hp.anafast(klm2_map * mask, input_kmap * mask, lmax=lmax)/fsky

rho12 = cross12 / np.sqrt(auto1 * auto2)
rho1k = cross1k / np.sqrt(autok * auto1)
rho2k = cross2k / np.sqrt(autok * auto2)
# Determinant of 2x2 correlation coefficient matrix rho with 1 as the diagonals and rho12 as the off diagonals
rho_det = 1 - rho12**2
# Invert rho
rhoinv11 = rhoinv22 = 1 / rho_det
rhoinv12 = rhoinv21 = -1*rho12 / rho_det

w1 = rhoinv11 * rho1k + rhoinv12 * rho2k
w2 = rhoinv22 * rho2k + rhoinv21 * rho1k
klm_combined = hp.almxfl(klm1,w1*np.sqrt(autok/auto1)) + hp.almxfl(klm2,w2*np.sqrt(autok/auto2))
klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
hp.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_qe_cib_phi_pp_tracer/klm_combined_cib_qe_pp_seed{idx}.alm', klm_combined)#, overwrite=True)


