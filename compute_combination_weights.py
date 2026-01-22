#!/usr/bin/env python
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
import argparse
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

def parse_path(yaml):
    config = utils.load_yaml(yaml)
    psname = config["pspec"]["psname"]

    dir_base = config["lensrec"]["dir_out"].format(
        rectype=config["lensrec"]["rectype"],
        runname=config["base"]["runname"],
        lmaxT=config["lensrec"]["lmaxT"],
        lminT=config["lensrec"]["lminT"],
        lminE=config["lensrec"]["lminE"],
        lminB=config["lensrec"]["lminB"],
        lmaxE=config["lensrec"]["lmaxE"],
        lmaxB=config["lensrec"]["lmaxB"],
        mmin=config["lensrec"]["mmin"],
    )

    if psname is not None and psname != "":
        dir_cls = dir_base + f"/clkk_polspice_{psname}_nops/"
    else:
        dir_cls = dir_base + f"/clkk_polspice_nops/"

    return dir_cls

# NOTE: change below
lmax = 2000
savedir = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/'
cliis = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clii_545ghz_patch{i}.dat")[:lmax+1,1] * (1e6/58.04)**2 for i in [0,1,2,3,5,6,7,8]])
clii = np.nanmean(cliis, axis=0)
cliks = np.array([np.loadtxt(f"/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/cls/cls_clik_545ghz_pr4kappa_patch{i}.dat")[:lmax+1,1] * (1e6/58.04) for i in [0,1,2,3,5,6,7,8]])
clik = np.nanmean(cliks, axis=0)
n0_std = utils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_mh.yaml'),498,'gmv','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
n0_prfhrd = utils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_prfhrd.yaml'),498,'gmvbhttprf','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
n0_pp = utils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_sqe.yaml'),498,'qpp','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
N0 = n0_std

def bin_interp_spectra(spectrum, li=np.arange(2,2001), average_window=81):
    spectrum_filtered = savgol_filter(spectrum, average_window, 0)
    cl_spline_mean = InterpolatedUnivariateSpline(np.arange(0, len(spectrum_filtered)), spectrum_filtered)
    cl_li = cl_spline_mean(li)
    return cl_li

li = np.arange(2,lmax+1)
nside = 2048
l = np.arange(lmax+1)
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
autok = slpp * (l*(l+1))**2/4

# SMOOTH
autok_spline = InterpolatedUnivariateSpline(ell, autok)
autok = autok_spline(li)
N0 = bin_interp_spectra(N0[:lmax+1], li)
clii = bin_interp_spectra(clii, li)
clik = bin_interp_spectra(clik, li)

auto1 = autok + N0
auto2 = clii
cross12 = clik
# noise doesn't correlate
cross1k = autok
cross2k = clik

rho12_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross12 / np.sqrt(auto1 * auto2)))
rho12 = rho12_spline(l)
rho1k_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross1k / np.sqrt(autok * auto1)))
rho1k = rho1k_spline(l)
rho2k_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross2k / np.sqrt(autok * auto2)))
rho2k = rho2k_spline(l)

# Determinant of 2x2 correlation coefficient matrix rho with 1 as the diagonals and rho12 as the off diagonals
rho_det = 1 - rho12**2
# Invert rho
rhoinv11 = rhoinv22 = 1 / rho_det
rhoinv12 = rhoinv21 = -1*rho12 / rho_det

w1 = rhoinv11 * rho1k + rhoinv12 * rho2k
w2 = rhoinv22 * rho2k + rhoinv21 * rho1k
kk_over_ii1_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(np.sqrt(autok/auto1)))
kk_over_ii1 = kk_over_ii1_spline(l)
kk_over_ii2_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(np.sqrt(autok/auto2)))
kk_over_ii2 = kk_over_ii2_spline(l)
save1 = w1*kk_over_ii1
save2 = w2*kk_over_ii2
#np.save(savedir+f'/klm1_weight', save1)
#np.save(savedir+f'/klm2_weight', save2)

# CHECK
save1_sim1 = np.load(savedir+'/klm1_weight_sim1.npy')
save2_sim1 = np.load(savedir+'/klm2_weight_sim1.npy')

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, save1_sim1/save1, label='w1, sim1/theory')
plt.plot(l, save2_sim1/save2, label='w2, sim1/theory')
plt.ylim(0,3)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

plt.clf()
plt.plot(l[2:],auto1,color='firebrick',label='auto1')
plt.plot(l[2:],auto2,color='darkblue',label='auto2')
plt.plot(l[2:],cross12,color='forestgreen',label='cross12 = cross2k')
plt.plot(l[2:],cross1k,color='darkorchid',label='cross1k = autok')
plt.plot(l[2:],auto1_sim1,color='pink',linestyle='--',label='auto1, sim1')
plt.plot(l[2:],auto2_sim1,color='cornflowerblue',linestyle='--',label='auto2, sim1')
plt.plot(l[2:],cross12_sim1,color='lightgreen',linestyle='--',label='cross12, sim1')
plt.plot(l[2:],cross1k_sim1,color='thistle',linestyle='--',label='cross1k, sim1')
plt.plot(l[2:],cross2k_sim1,color='mediumturquoise',linestyle='--',label='cross2k, sim1')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')
'''
#=============================================================================#
# Once below is run for 499 sims, comment out everything below and comment in:
#savedir = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/'
#idxs = np.arange(499)+1
#save1 = 0
#save2 = 0
#for idx in idxs:
#    save1 += np.load(savedir+f'/klm1_weight_{idx}.dat')
#    save2 += np.load(savedir+f'/klm2_weight_{idx}.dat')
#save1 /= len(idxs)
#save2 /= len(idxs)
#np.save(savedir+f'/klm1_weight_avg.dat', save1)
#np.save(savedir+f'/klm2_weight_avg.dat', save2)
#=============================================================================#

parser = argparse.ArgumentParser()
parser.add_argument('idx'       , default=None, type=int, help='idx')
args = parser.parse_args()
idx = args.idx
print(idx)

# GET AND SAVE WEIGHTS FIRST
# NOTE: change below
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
savedir = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/'
cib_tracer = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_tracers/pr3_cib_pr4_kappa_tracer/cib_tracer_seed{idx}.fits'

def tp2rd(tht, phi):
    ra=phi/np.pi*180.0
    dec=((tht*-1)+np.pi/2.0)/np.pi*180.
    return ra,dec

def rd2tp(ra, dec):
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

def bin_interp_spectra(spectrum, li=np.arange(2,2001), average_window=81):
    spectrum_filtered = savgol_filter(spectrum, average_window, 0)
    cl_spline_mean = InterpolatedUnivariateSpline(np.arange(0, len(spectrum_filtered)), spectrum_filtered)
    cl_li = cl_spline_mean(li)
    return cl_li

# https://arxiv.org/pdf/2212.07420 equations 89 - 92
# https://arxiv.org/pdf/1705.02332 equations 7 - 9
btmp = bt.btemplate(yaml_file,combined_tracer=True)
lmax = btmp.lmax_b
l = np.arange(lmax+1)
li = np.arange(2,lmax+1)
nside = btmp.nside
mask = hp.read_map(btmp.maskfname)
fsky = np.sum(mask**2)/mask.size

# Get reconstructed 2019/2020 analysis phi tracer
klm1 = btmp.get_debiased_klm(idx)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)
klm1_map = hp.alm2map(klm1, nside)
auto1 = hp.anafast(klm1_map * mask, lmax=lmax)/fsky
# Smoothing
auto1 = bin_interp_spectra(auto1, li)

# Get CIB-based phi tracer
klm2_map = hp.read_map(cib_tracer)
auto2 = hp.anafast(klm2_map * mask, lmax=lmax)/fsky
cross12 = hp.anafast(klm1_map * mask, klm2_map * mask, lmax=lmax)/fsky
auto2 = bin_interp_spectra(auto2, li)
cross12 = bin_interp_spectra(cross12, li)
# TODO: there will be ringing here...
klm2 = hp.map2alm(klm2_map, lmax=lmax)

# Get input kappa
# Use fiducial for kappa auto for less noise
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
autok = slpp * (l*(l+1))**2/4
autok_spline = InterpolatedUnivariateSpline(ell, autok)
autok = autok_spline(li)
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
cross1k = bin_interp_spectra(cross1k, li)
cross2k = bin_interp_spectra(cross2k, li)

rho12_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross12 / np.sqrt(auto1 * auto2)))
rho12 = rho12_spline(l)
rho1k_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross1k / np.sqrt(autok * auto1)))
rho1k = rho1k_spline(l)
rho2k_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(cross2k / np.sqrt(autok * auto2)))
rho2k = rho2k_spline(l)
# Determinant of 2x2 correlation coefficient matrix rho with 1 as the diagonals and rho12 as the off diagonals
rho_det = 1 - rho12**2
# Invert rho
rhoinv11 = rhoinv22 = 1 / rho_det
rhoinv12 = rhoinv21 = -1*rho12 / rho_det

w1 = rhoinv11 * rho1k + rhoinv12 * rho2k
w2 = rhoinv22 * rho2k + rhoinv21 * rho1k
kk_over_ii1_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(np.sqrt(autok/auto1)))
kk_over_ii1 = kk_over_ii1_spline(l)
kk_over_ii2_spline = InterpolatedUnivariateSpline(li, np.nan_to_num(np.sqrt(autok/auto2)))
kk_over_ii2 = kk_over_ii2_spline(l)
#klm_combined = hp.almxfl(klm1,w1*kk_over_ii1) + hp.almxfl(klm2,w2*kk_over_ii2)
#klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
save1 = w1*kk_over_ii1
save2 = w2*kk_over_ii2
np.save(savedir+f'/klm1_weight_sim{idx}', save1)
np.save(savedir+f'/klm2_weight_sim{idx}', save2)
'''
