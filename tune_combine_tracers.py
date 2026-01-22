#!/usr/bin/env python
import numpy as np
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

#parser = argparse.ArgumentParser()
#parser.add_argument('idx'       , default=None, type=int, help='idx')
#args = parser.parse_args()
#idx = args.idx
#print(idx)

# NOTE: change below
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
btmp = bt.btemplate(yaml_file,combined_tracer=True)
lmax = btmp.lmax_b
l = np.arange(lmax+1)
nside = btmp.nside
weightsdir = btmp.combined_tracer_weights_dir
weights1 = np.load(weightsdir+'/klm1_weight.npy')
weights2 = np.load(weightsdir+'/klm2_weight.npy')
cib_tracer_dir = btmp.cib_tracer_dir

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

#====== DRAW FOR SOME PRIOR FOR WEIGHTS (e.g. A_i ~ N(1, sigma(A)^2)) ======#
# GRID SEARCH (slower) OR MINIMIZER
# For a grid search, we step through pre-determined values of weights (start with a coarse grid and then make it finer)
# For a minimizer, you can use scipy's optimize class
# Try grid search first...
def random_grid_search(weights1_base,weights2_base,bin_edges,btmp,klm1,klm2,
                       n_draws=200,sigma_A1=0.2,sigma_A2=0.2,
                       tune_A1=True,tune_A2=True,seed=0):

    lmax = btmp.lmax_b
    rng = np.random.default_rng(seed)
    nbins = len(bin_edges) - 1
    best = {
        "a_pow": -np.inf,
        "A1_bins": None,
        "A2_bins": None,
        "weights1_L": None,
        "weights2_L": None,
    }

    # baseline (all ones)
    A1_center = np.ones(nbins)
    A2_center = np.ones(nbins)

    for t in range(n_draws):
        # Step 1: draw from prior around 1
        if tune_A1:
            # draw vector of length nbins where each element is an independent
            # Gaussian number A_i ~ N(1,sigma(A)^2)
            # samples from a normal distribution: loc=mean, scale=standard deviation, size=output shape
            A1_bins = rng.normal(loc=1.0, scale=sigma_A1, size=nbins)
        else:
            A1_bins = A1_center.copy()

        if tune_A2:
            A2_bins = rng.normal(loc=1.0, scale=sigma_A2, size=nbins)
        else:
            A2_bins = A2_center.copy()

        # Expand to full L weights
        weights1_L = apply_bin_scalings_to_weights(weights1_base, bin_edges, A1_bins, lmax)
        weights2_L = apply_bin_scalings_to_weights(weights2_base, bin_edges, A2_bins, lmax)

        # Evaluate
        a_pow = score(weights1_L=weights1_L,weights2_L=weights2_L,btmp=btmp,klm1=klm1, klm2=klm2)

        if a_pow > best["a_pow"]:
            best.update(
                a_pow=a_pow,
                A1_bins=A1_bins.copy(),
                A2_bins=A2_bins.copy(),
                weights1_L=weights1_L.copy(),
                weights2_L=weights2_L.copy(),
            )
            print(f"[best @ draw {t}] a_pow={a_pow:.6e}")

        if t % 5 == 0:
            print(f"[progress] draw {t}/{n_draws} current a_pow={a_pow:.6e} best={best['a_pow']:.6e}", flush=True)

        if t % 10 == 0:
            np.save(btmp.combined_tracer_weights_dir + f"/checkpoint_klm1_weight_tuned.npy", best["weights1_L"])
            np.save(btmp.combined_tracer_weights_dir + f"/checkpoint_klm2_weight_tuned.npy", best["weights2_L"])

    return best

def evaluate_candidate(weights1_L,weights2_L,idx,btmp,klm1,klm2):
    klm_combined = hp.almxfl(klm1,weights1_L) + hp.almxfl(klm2,weights2_L)
    klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
    hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed{idx}_tuned.alm', klm_combined, overwrite=True)

    #========== CALCULATE LT AUTO ==========#
    a, c, a_in = btmp.get_masked_spec(idx,overwrite=True,savefile=False)

    #=== MEASURE POWER IN LT AUTO (sum across ell = [50,200]) AND MAXIMIZE ===#
    a_pow = np.mean(a[50:201])
    return a_pow

def score(weights1_L,weights2_L,btmp,klm1,klm2):
    vals = []
    idxs = np.arange(10)+1
    for idx in idxs:
        ## Get reconstructed 2019/2020 analysis phi tracer FOR EACH SIM
        #k1 = btmp.get_debiased_klm(idx)
        #k1 = utils.reduce_lmax(k1, lmax=lmax)
        ## Get CIB-based phi tracer
        #klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{idx}.fits')
        #k2 = hp.map2alm(klm2_map, lmax=lmax)
        k1 = klm1[idx-1,:]
        k2 = klm2[idx-1,:]
        vals.append(evaluate_candidate(weights1_L=weights1_L,weights2_L=weights2_L,
                                       idx=idx,btmp=btmp,klm1=k1,klm2=k2))
    return np.mean(vals)

#=============================================================================#

#========== COMPUTE COMBINED TRACER ==========#
cib_tracer_dir = btmp.cib_tracer_dir
idxs = np.arange(10)+1
tmp = btmp.get_debiased_klm(1)
klm1 = np.zeros((len(idxs),len(tmp)),dtype=np.complex_)
klm2 = np.zeros((len(idxs),len(tmp)),dtype=np.complex_)
for idx in idxs:
    print(f'loading idx: {idx}',flush=True)
    # Get reconstructed 2019/2020 analysis phi tracer FOR EACH SIM
    klm1[idx-1,:] = btmp.get_debiased_klm(idx)
    klm1[idx-1,:] = utils.reduce_lmax(klm1[idx-1,:], lmax=lmax)
    # Get CIB-based phi tracer
    klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{idx}.fits')
    klm2[idx-1,:] = hp.map2alm(klm2_map, lmax=lmax)

# choose bins (keep small)
bin_edges = np.array([0, 50, 100, 500, 1000, 1500, 2001])

best = random_grid_search(weights1_base=weights1,weights2_base=weights2,
                          btmp=btmp,klm1=klm1, klm2=klm2,
                          bin_edges=bin_edges,n_draws=100,#n_draws=300,
                          sigma_A1=0.2, #0.05, # usually keep QE scalings tight if you even tune them
                          sigma_A2=0.2, #0.30, # allow more freedom on CIB
                          tune_A1=True,tune_A2=True,seed=1,)

# Save best weights
np.save(weightsdir+f'/klm1_weight_tuned.npy', best["weights1_L"])
np.save(weightsdir+f'/klm2_weight_tuned.npy', best["weights2_L"])
np.save(weightsdir+f'/A1_bins_klm1_weight_tuned.npy', best["A1_bins"])
np.save(weightsdir+f'/A2_bins_klm2_weight_tuned.npy', best["A2_bins"])
print("best a_pow:", best["a_pow"])
print("best A1 bins:", best["A1_bins"])
print("best A2 bins:", best["A2_bins"])
for idx in idxs:
    ## Get reconstructed 2019/2020 analysis phi tracer FOR EACH SIM
    #k1 = btmp.get_debiased_klm(idx)
    #k1 = utils.reduce_lmax(k1, lmax=lmax)
    ## Get CIB-based phi tracer
    #klm2_map = hp.read_map(cib_tracer_dir + f'/cib_tracer_seed{idx}.fits')
    #k2 = hp.map2alm(klm2_map, lmax=lmax)
    k1 = klm1[idx-1,:]
    k2 = klm2[idx-1,:]
    klm_combined = hp.almxfl(k1,best["weights1_L"]) + hp.almxfl(k2,best["weights2_L"])
    klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
    hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed{idx}_tuned.alm', klm_combined, overwrite=True)
    a, c, a_in = btmp.get_masked_spec(idx,overwrite=True,savefile=True)
    print('SAVED best combined tracer and btemplate for:', idx)

plt.figure(0)
plt.clf()
plt.plot(l, weights1, color='firebrick', label='w1')
plt.plot(l, weights2, color='darkblue', label='w2')
plt.plot(l, best["weights1_L"], color='pink', linestyle='--', label='tuned w1')
plt.plot(l, best["weights2_L"], color='cornflowerblue', linestyle='--', label='tuned w2')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

