#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
import argparse
import time
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

# GET AND SAVE WEIGHTS FIRST

parser = argparse.ArgumentParser()
parser.add_argument('yaml', default=None, type=str, help='yaml')
parser.add_argument('idx', default=None, type=int, help='idx')
parser.add_argument('seed', default=None, type=int, help='seed')
parser.add_argument("--use_10_sims",  default=False, action="store_true", dest="use_10_sims", help="use_10_sims")
parser.add_argument("--no_seed",  default=False, action="store_true", dest="no_seed", help="no_seed")
args = parser.parse_args()
yaml_file = args.yaml
idx = args.idx
seed = args.seed
use_10_sims = args.use_10_sims
no_seed = args.no_seed
if no_seed:
    seed = None
if use_10_sims:
    idx = None

nbins = 6
btmp = bt.btemplate(yaml_file,combined_tracer=True)
lmax = btmp.lmax_b
l = np.arange(lmax+1)
nside = btmp.nside
weightsdir = btmp.combined_tracer_weights_dir
rhos1 = np.load(weightsdir+'/klm1_weight_rhos.npy')
rhos2 = np.load(weightsdir+'/klm2_weight_rhos.npy')
sqrt_kk_over_ii1 = np.load(weightsdir+'/klm1_weight_sqrt_kk_over_ii.npy')
sqrt_kk_over_ii2 = np.load(weightsdir+'/klm2_weight_sqrt_kk_over_ii.npy')
cib_tracer_dir = btmp.cib_tracer_dir
assert np.all(rhos1 >= 0)
assert np.all(rhos2 >= 0)

#========== SECTION WEIGHTS IN TO L BINS ==========#
def expand_bins_to_L(bin_edges, A, lmax):
    """
    bin_edges: array of length nbins+1, inclusive edges in L
    A: array length nbins, piecewise constant A(L) in each bin
    returns A_L array length lmax+1
    """
    ret = np.zeros(lmax + 1, dtype=float)
    for i in range(len(A)):
        L0, L1 = int(bin_edges[i]), int(bin_edges[i+1])
        ret[L0:L1] = A[i]
    return ret

def apply_bin_scalings_to_weights(weights_base, bin_edges, A, lmax):
    """
    Return weights(L) = A(L) * weights_base(L).
    """
    A_L = expand_bins_to_L(bin_edges, A, lmax)
    out = weights_base.copy()
    out[:lmax+1] *= A_L
    return out

def weights_from_xbins(x_bins, rhos1_base, rhos2_base, bin_edges, lmax, sqrt1, sqrt2):
    # expand x to L
    x_L = expand_bins_to_L(bin_edges, x_bins, lmax)
    # ratio R = exp(x)
    R_L = np.exp(x_L)

    r1 = np.maximum(0.0, rhos1_base * R_L)
    r2 = np.maximum(0.0, rhos2_base)

    den = r1 + r2
    den[den == 0] = 1.0
    r1 /= den
    r2 /= den

    w1 = r1 * sqrt1
    w2 = r2 * sqrt2
    return w1, w2

def obj(x_bins, *args):
    (rhos1_base, rhos2_base, bin_edges, btmp, klm1, klm2, sqrt1, sqrt2, idxs) = args
    w1, w2 = weights_from_xbins(x_bins, rhos1_base, rhos2_base, bin_edges, btmp.lmax_b, sqrt1, sqrt2)
    s = score(weights1_L=w1, weights2_L=w2, btmp=btmp, klm1=klm1, klm2=klm2, idxs=idxs)
    return -s  # maximize s

def evaluate_candidate(weights1_L,weights2_L,i,btmp,klm1,klm2):
    #========== COMPUTE COMBINED TRACER ==========#
    klm_combined = hp.almxfl(klm1,weights1_L) + hp.almxfl(klm2,weights2_L)
    klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
    hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed{i}_tuned.alm', klm_combined, overwrite=True)

    #========== CALCULATE LT AUTO ==========#
    a, c, a_in = btmp.get_masked_spec(i,recompute=True,savefile=False)

    #=== MEASURE POWER IN LT AUTO (sum across ell = [30,200]) AND MAXIMIZE ===#
    a_pow = np.mean(a[30:201])
    return a_pow

def score(weights1_L,weights2_L,btmp,klm1,klm2,idxs):
    vals = []
    if idxs[0] == 0 or idxs[0] == 5001:
        # FOR DATA
        k1 = klm1
        k2 = klm2
        vals.append(evaluate_candidate(weights1_L=weights1_L,weights2_L=weights2_L,
                                       i=idxs[0],btmp=btmp,klm1=k1,klm2=k2,))
    else:
        for i in idxs:
            k1 = klm1[i-idxs[0],:]
            k2 = klm2[i-idxs[0],:]
            vals.append(evaluate_candidate(weights1_L=weights1_L,weights2_L=weights2_L,
                                           i=i,btmp=btmp,klm1=k1,klm2=k2,))
    return np.mean(vals)

#=============================================================================#

cib_tracer_dir = btmp.cib_tracer_dir
if idx is not None:
    idxs = [idx]
else:
    idxs = np.arange(10)+1
print('idxs:', idxs)
print('seed:', seed)
# Get reconstructed 2019/2020 analysis phi tracer FOR DATA
if idx == 0:
    klm1 = btmp.get_debiased_klm(0)
    klm1 = utils.reduce_lmax(klm1, lmax=lmax)
    # Get CIB-based phi tracer
    klm2_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/planck_pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits")
    rot = hp.Rotator(coord=['G','C'])
    klm2_map = rot.rotate_map_pixel(klm2_map)
    mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits")
    klm2_map *= mask * 1e6 / 58.04
    klm2 = hp.map2alm(klm2_map, lmax=lmax)
elif idx == 5001:
    # AGORA
    klm1 = btmp.get_debiased_klm(idx)
    klm1 = utils.reduce_lmax(klm1, lmax=lmax)
    # Get CIB
    klm2_map = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_len_mag_cibmap_planck_545ghz_nside2048.fits")
    mask = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits")
    klm2_map *= mask * (1/58.04)
    klm2 = hp.map2alm(klm2_map, lmax=lmax)
else:
    tmp = btmp.get_debiased_klm(1)
    klm1 = np.zeros((len(idxs),len(tmp)),dtype=np.complex_)
    klm2 = np.zeros((len(idxs),len(tmp)),dtype=np.complex_)
    for i in idxs:
        print(f'loading idx: {i}',flush=True)
        # Get reconstructed 2019/2020 analysis phi tracer FOR EACH SIM
        klm1[i-idxs[0],:] = btmp.get_debiased_klm(i)
        klm1[i-idxs[0],:] = utils.reduce_lmax(klm1[i-idxs[0],:], lmax=lmax)
        # Get CIB-based phi tracer
        klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{i}.fits')
        klm2[i-idxs[0],:] = hp.map2alm(klm2_map, lmax=lmax)

# choose bins (keep small)
bin_edges = np.array([0, 50, 100, 500, 1000, 1500, 2001])

# initial guess: no change (ratio=1)
x0 = np.zeros(nbins)

# bounds in log-ratio space
bounds = [(-1.0, 1.0)] * nbins

args_obj = (rhos1, rhos2, bin_edges, btmp, klm1, klm2, sqrt_kk_over_ii1, sqrt_kk_over_ii2, idxs)

res = minimize(obj, x0, args=args_obj, method="Powell", bounds=bounds,
               options={"maxiter": 40, "disp": True})

best_x = res.x
best_w1, best_w2 = weights_from_xbins(best_x, rhos1, rhos2, bin_edges, lmax, sqrt_kk_over_ii1, sqrt_kk_over_ii2)

# Save best weights
if idx is not None and seed is not None:
    np.save(weightsdir+f'/klm1_weight_tuned_sim{idx}_seed{seed}.npy', best_w1)
    np.save(weightsdir+f'/klm2_weight_tuned_sim{idx}_seed{seed}.npy', best_w2)
    np.save(weightsdir+f'/A1_bins_klm1_weight_tuned_sim{idx}_seed{seed}.npy', best_w1)
    np.save(weightsdir+f'/A2_bins_klm2_weight_tuned_sim{idx}_seed{seed}.npy', best_w2)
elif idx is not None:
    np.save(weightsdir+f'/klm1_weight_tuned_sim{idx}.npy', best_w1)
    np.save(weightsdir+f'/klm2_weight_tuned_sim{idx}.npy', best_w2)
    np.save(weightsdir+f'/A1_bins_klm1_weight_tuned_sim{idx}.npy', best_w1)
    np.save(weightsdir+f'/A2_bins_klm2_weight_tuned_sim{idx}.npy', best_w2)
elif seed is not None:
    np.save(weightsdir+f'/klm1_weight_tuned_sims1to10_seed{seed}.npy', best_w1)
    np.save(weightsdir+f'/klm2_weight_tuned_sims1to10_seed{seed}.npy', best_w2)
    np.save(weightsdir+f'/A1_bins_klm1_weight_tuned_sims1to10_seed{seed}.npy', best_w1)
    np.save(weightsdir+f'/A2_bins_klm2_weight_tuned_sims1to10_seed{seed}.npy', best_w2)
else:
    np.save(weightsdir+f'/klm1_weight_tuned_sims1to10.npy', best_w1)
    np.save(weightsdir+f'/klm2_weight_tuned_sims1to10.npy', best_w2)
    np.save(weightsdir+f'/A1_bins_klm1_weight_tuned_sims1to10.npy', best_w1)
    np.save(weightsdir+f'/A2_bins_klm2_weight_tuned_sims1to10.npy', best_w2)

print("best x bins:", best_x)
print("best w1:", best_w1)
print("best w2:", best_w2)

for i in idxs:
    if i == 0 or i > 5000:
        k1 = klm1
        k2 = klm2
    else:
        k1 = klm1[i-idxs[0],:]
        k2 = klm2[i-idxs[0],:]
    klm_combined = hp.almxfl(k1,best_w1) + hp.almxfl(k2,best_w2)
    klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
    hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed{i}_tuned.alm', klm_combined, overwrite=True)
    a, c, a_in = btmp.get_masked_spec(i,recompute=True,savefile=True)
    print('SAVED best combined tracer and btemplate for:', i)

plt.figure(0)
plt.clf()
plt.plot(l, rhos1, color='firebrick', label='rhos1')
plt.plot(l, rhos2, color='darkblue', label='rhos2')
plt.plot(l, best_w1/sqrt_kk_over_ii1, color='pink', linestyle='--', label='tuned rhos1')
plt.plot(l, best_w2/sqrt_kk_over_ii2, color='cornflowerblue', linestyle='--', label='tuned rhos2')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.ylim(1e-2,2)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/rhos_weights_tuned.png')

