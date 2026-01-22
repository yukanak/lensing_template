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

# GET AND SAVE WEIGHTS FIRST

parser = argparse.ArgumentParser()
parser.add_argument('idx'       , default=None, type=int, help='idx')
args = parser.parse_args()
idx = args.idx
print(idx)

# https://arxiv.org/pdf/2212.07420 equations 89 - 92
# https://arxiv.org/pdf/1705.02332 equations 7 - 9
# NOTE: change below
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
btmp = bt.btemplate(yaml_file,combined_tracer=True)
lmax = btmp.lmax_b
l = np.arange(lmax+1)
li = np.arange(2,lmax+1)
nside = btmp.nside
weightsdir = btmp.combined_tracer_weights_dir
weights1 = np.load(weightsdir+'/klm1_weight.npy')
weights2 = np.load(weightsdir+'/klm2_weight.npy')
cib_tracer_dir = btmp.cib_tracer_dir

# Get reconstructed 2019/2020 analysis phi tracer
klm1 = btmp.get_debiased_klm(idx)
klm1 = utils.reduce_lmax(klm1, lmax=lmax)

# Get CIB-based phi tracer
#klm2 = hp.read_alm(cib_tracer_dir + f'/cib_klm_seed{idx}.alm')
klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{idx}.fits')
# TODO: there will be ringing here...
klm2 = hp.map2alm(klm2_map, lmax=lmax)

klm_combined = hp.almxfl(klm1,weights1) + hp.almxfl(klm2,weights2)
klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed{idx}.alm', klm_combined)#, overwrite=True)

