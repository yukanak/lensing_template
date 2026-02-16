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

# THIS DUMMY TRACER (OR I GUESS TWO TRACER TYPES) I1 and I2 HAVE SAME SIGNAL
# JUST DIFFERENT NOISE SPECTRA
# SIGNAL PART I SET TO INPUT KAPPA SIM 1
# FOR ONE OF THEM I CAN JUST USE N0 AS NOISE SPECTRUM
# THEN THE OTHER ONE I HAVE A SLIGHTLY DIFFERENT NOISE SPECTRUM

parser = argparse.ArgumentParser()
parser.add_argument('idx'       , default=None, type=int, help='idx')
args = parser.parse_args()
idx = args.idx
print(idx)

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

dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_tracers/dummy_tracer/'
lmax = 2000
nside = 2048
l = np.arange(lmax+1)
li = np.arange(2,lmax+1)
n0_std = utils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_mh.yaml'),498,'gmv','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
N0 = bin_interp_spectra(n0_std[:lmax+1] * (l*(l+1))**2/4, li)
# Load SIM 1 plm and convert to klm
input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_4096/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed1_lmax4096.alm')
input_plm = utils.reduce_lmax(input_plm, lmax=lmax)
ell, m = hp.Alm.getlm(lmax)
fac = np.zeros_like(ell, dtype=float)
fac[ell>=2] = 0.5 * ell[ell>=2]*(ell[ell>=2]+1.0)
input_klm = hp.almxfl(input_plm, fac)
input_kmap = hp.alm2map(input_klm, nside=nside, lmax=lmax)
# signal
I_sig_map = input_kmap
# noise
nl_spline = InterpolatedUnivariateSpline(li, N0)
nl = nl_spline(l)
I_noi = hp.synalm(nl, lmax=lmax, new=True)
I_noi_map = hp.alm2map(I_noi, nside=nside, lmax=lmax)
# total
I_map = I_sig_map + I_noi_map
# SAVE CIB-BASED PHI TRACER AS JUST THE I, NO WIENER FILTERING HERE
hp.write_map(dir_out+f'tracer_type1_seed{idx}.fits', I_map, overwrite=True)

# DIFFERENT NOISE SPEC
I_noi2 = hp.synalm(10*nl, lmax=lmax, new=True)
I_noi_map2 = hp.alm2map(I_noi2, nside=nside, lmax=lmax)
# total
I_map2 = I_sig_map + I_noi_map2
# SAVE CIB-BASED PHI TRACER AS JUST THE I, NO WIENER FILTERING HERE
hp.write_map(dir_out+f'tracer_type2_seed{idx}.fits', I_map2, overwrite=True)
