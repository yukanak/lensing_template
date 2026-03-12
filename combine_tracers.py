#!/usr/bin/env python
import numpy as np
from pathlib import Path
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
import argparse
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

# GET AND SAVE WEIGHTS FIRST

parser = argparse.ArgumentParser()
parser.add_argument('yaml', default=None, type=str, help='yaml') 
parser.add_argument('idx'       , default=None, type=int, help='idx')
parser.add_argument('Lmin_cib'       , default=None, type=int, help='Lmin_cib')
args = parser.parse_args()
yaml_file = args.yaml
idx = args.idx
Lmin_cib = args.Lmin_cib
print(idx)

# https://arxiv.org/pdf/2212.07420 equations 89 - 92
# https://arxiv.org/pdf/1705.02332 equations 7 - 9
btmp = bt.btemplate(yaml_file,combined_tracer=True)
lmax = btmp.lmax_b
l = np.arange(lmax+1)
li = np.arange(2,lmax+1)
nside = btmp.nside
weightsdir = btmp.combined_tracer_weights_dir
#weights1 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/lenz_cib_pr4_kappa_standard/weights_from_sims/klm1_weight_avg.npy')
#weights2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/lenz_cib_pr4_kappa_standard/weights_from_sims/klm2_weight_avg.npy')
#weights1 = np.load(weightsdir+'/klm1_weight.npy')
#weights2 = np.load(weightsdir+'/klm2_weight.npy')
weights1 = np.load(weightsdir+f'/klm1_weight_ciblmin{Lmin_cib}.npy')
weights2 = np.load(weightsdir+f'/klm2_weight_ciblmin{Lmin_cib}.npy')
cib_tracer_dir = btmp.cib_tracer_dir

# Get reconstructed 2019/2020 analysis phi tracer
klm1 = btmp.get_debiased_klm(idx)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)

# Get CIB-based phi tracer
#klm2 = hp.read_alm(cib_tracer_dir + f'/cib_klm_seed{idx}.alm')
if idx != 0 and idx < 5000:
    klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{idx}.fits')
elif idx == 0:
    # DATA!
    # NOTE: hard-coded for PR3 GNILC so do NOT use for Lenz et al. etc.!
    mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits")
    klm2_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits")
    rot = hp.Rotator(coord=['G','C'])
    klm2_map = rot.rotate_map_pixel(klm2_map)
    klm2_map *= mask * 1e6 / 58.04
else:
    # AGORA!
    mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits")
    klm2_map = hp.read_map(f"/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/cib/agora_len_mag_cibmap_planck_545ghz_nside2048_rotated_{idx}.alm")
    klm2_map *= mask * (1/58.04) 
# TODO: there will be ringing here...
klm2 = hp.map2alm(klm2_map, lmax=lmax)

klm_combined = hp.almxfl(klm1,weights1) + hp.almxfl(klm2,weights2)
klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
Path(btmp.dir_combined_tracer+f'/ciblmin{Lmin_cib}/').mkdir(parents=True, exist_ok=True)
hp.write_alm(btmp.dir_combined_tracer+f'/ciblmin{Lmin_cib}/klm_combined_cib_qe_seed{idx}.alm', klm_combined)#, overwrite=True)

