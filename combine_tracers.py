import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

# https://arxiv.org/pdf/2212.07420 equations 89 - 92
# https://arxiv.org/pdf/1705.02332 equations 7 - 9
idx = 1
yaml_file = 'bt_gmv3500.yaml'
btmp = bt.btemplate(yaml_file)
#lmax = btmp.lmax_b
lmax = 1500
nside = btmp.nside
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size
l = np.arange(lmax_b+1)

# Get reconstructed 2019/2020 analysis phi tracer
klm1 = btmp.get_debiased_klm(idx)
klm1_map = hp.alm2map(klm1, nside, lmax=lmax)
auto1 = hp.anafast(klm1_map * mask)/fsky

# Get CIB-based phi tracer
klm2 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_phi_tracer/cib_klm_seed{idx}.alm')
klm2_map = hp.alm2map(klm2, nside, lmax=lmax)
auto2 = hp.anafast(klm2_map * mask)/fsky
cross12 = hp.anafast(klm1_map * mask, klm2_map * mask)/fsky

# Get input kappa
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
autok = slpp * (l*(l+1))**2/4 #TODO mask or fsky?
#autok = hp.anafast(kmap_fid * mask)/fsky
#cross1k = hp.anafast(klm1_map * mask, kmap_fid * mask)/fsky
#cross2k = hp.anafast(klm2_map * mask, kmap_fid * mask)/fsky

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
klm_combined = w1 * klm1 * np.sqrt(autok/auto1) + w2 * klm2 * np.sqrt(autok/auto2)



