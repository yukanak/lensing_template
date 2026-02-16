import numpy as np
import healpy as hp
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

#yaml_file = 'bt_gmv3500_combined_lenz.yaml'
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
btmp = bt.btemplate(yaml_file, combined_tracer=True)
yaml_file = 'bt_gmv3500_combined_agora545_cib.yaml'
btmp_agora = bt.btemplate(yaml_file, combined_tracer=True)
lmax = 2000
ell = np.arange(lmax+1)
l = np.arange(lmax+1)
lbins = np.logspace(np.log10(30),np.log10(2000),20)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(np.arange(6144), lbins)
# NO TUNING
auto = 0
cross = 0
auto_in = 0
for i in np.arange(10)+1:
    fname = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_pr3_cib_pr4_kappa_standard/btmpl_specs_{i:04d}.npz'
    tmp = np.load(fname)
    auto += tmp['auto']
    cross += tmp['cross']
    auto_in += tmp['auto_in']
auto /= 10
cross /= 10
auto_in /= 10
#fname = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_pr3_cib_pr4_kappa_standard/btmpl_specs_0000.npz'
#fname = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_agora545_cib_standard/btmpl_specs_5001_assumegalactic.npz'
fname = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_agora545_cib_standard/btmpl_specs_5001.npz'
tmp   = np.load(fname)
auto_data  = tmp['auto']
cross_data = tmp['cross']
auto_in_data = tmp['auto_in']
# WITH TUNING
auto1, cross1, auto_in1 = btmp.get_masked_spec(1,recompute=False)
#auto1_data, _, _ = btmp.get_masked_spec(0,recompute=False)
auto1_data, cross1_data, auto_in1_data = btmp_agora.get_masked_spec(5001,recompute=False)

# BIN
auto = np.array([auto[digitized == i].mean() for i in range(1, len(lbins))])
cross = np.array([cross[digitized == i].mean() for i in range(1, len(lbins))])
auto_in = np.array([auto_in[digitized == i].mean() for i in range(1, len(lbins))])
auto1 = np.array([auto1[digitized == i].mean() for i in range(1, len(lbins))])
cross1 = np.array([cross1[digitized == i].mean() for i in range(1, len(lbins))])
auto_in1 = np.array([auto_in1[digitized == i].mean() for i in range(1, len(lbins))])
auto_data = np.array([auto_data[digitized == i].mean() for i in range(1, len(lbins))])
cross_data = np.array([cross_data[digitized == i].mean() for i in range(1, len(lbins))])
auto_in_data = np.array([auto_in_data[digitized == i].mean() for i in range(1, len(lbins))])
auto1_data = np.array([auto1_data[digitized == i].mean() for i in range(1, len(lbins))])
cross1_data = np.array([cross1_data[digitized == i].mean() for i in range(1, len(lbins))])
auto_in1_data = np.array([auto_in1_data[digitized == i].mean() for i in range(1, len(lbins))])

# Plot
plt.figure(0)
plt.clf()
plt.plot(bin_centers, auto[:lmax+1], color='firebrick', linestyle='-', label=f'btemplate auto, avg sims 1-10')
plt.plot(bin_centers, cross[:lmax+1], color='darkblue', linestyle='-', label=f'btemplate x input B cross, avg sims 1-10')
#plt.plot(bin_centers, auto_data[:lmax+1], color='darkorange', linestyle='-', label=f'btemplate auto, Agora 5001')
#plt.plot(bin_centers, cross_data[:lmax+1], color='slateblue', linestyle='-', label=f'btemplate x input B cross, Agora 5001')
#plt.plot(bin_centers, auto_data[:lmax+1], color='mediumorchid', linestyle='-', label=f'btemplate auto, DATA')

plt.plot(bin_centers, auto_in[:lmax+1], color='forestgreen', linestyle='-', label=f'input B auto, avg sims 1-10')
#plt.plot(bin_centers, auto_in_data[:lmax+1], color='olive', linestyle='-', label=f'input B auto, Agora 5001')

plt.plot(bin_centers, auto1[:lmax+1], color='pink', linestyle='--', label='btemplate auto, avg sims 1-10, TUNED')
plt.plot(bin_centers, cross1[:lmax+1], color='powderblue', linestyle='--', label='btemplate x input B cross, avg sims 1-10, TUNED')
#plt.plot(bin_centers, auto1_data[:lmax+1], color='orange', linestyle='--', label='btemplate auto, Agora 5001, TUNED')
#plt.plot(bin_centers, cross1_data[:lmax+1], color='mediumslateblue', linestyle='--', label=f'btemplate x input B cross, Agora 5001, TUNED')
#plt.plot(bin_centers, auto1_data[:lmax+1], color='thistle', linestyle='--', label='btemplate auto, DATA, TUNED')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(3e-7,3e-6)
plt.legend(loc='lower left', fontsize='small')
plt.title(f'btemplate check')
plt.ylabel("$C_\ell^{BB}$")
plt.xlabel('$\ell$')
plt.savefig('figs/btemplate_check_spec.png',bbox_inches='tight')

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(bin_centers, (auto1/auto), color='firebrick', linestyle='--', alpha=0.8, label=f'btemplate auto, avg sims 1-10 TUNED / NOT TUNED')
plt.plot(bin_centers, (cross1/cross), color='darkblue', linestyle='--', alpha=0.8, label=f'btemplate x input B cross, avg sims 1-10 TUNED / NOT TUNED')
#plt.plot(bin_centers, (auto1_data/auto_data), color='darkorange', linestyle='--', alpha=0.8, label=f'btemplate auto, Agora 5001 TUNED / NOT TUNED')
#plt.plot(bin_centers, (cross1_data/cross_data), color='slateblue', linestyle='--', alpha=0.8, label=f'btemplate x input B cross, Agora 5001 TUNED / NOT TUNED')
#plt.plot(bin_centers, (auto1_data/auto_data), color='mediumorchid', linestyle='--', alpha=0.8, label=f'btemplate auto, data TUNED / NOT TUNED')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0.95,1.3)
plt.legend(loc='lower left', fontsize='small')
plt.title(f'btemplate check')
plt.xlabel('$\ell$')
plt.savefig('figs/btemplate_check_ratio.png',bbox_inches='tight')

weightsdir = btmp.combined_tracer_weights_dir
#weights1 = np.load(weightsdir+'/klm1_weight.npy')
#weights2 = np.load(weightsdir+'/klm2_weight.npy')
#w1 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm1_weight_tuned.npy')
#w2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm2_weight_tuned.npy')
#w1 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/klm1_weight_tuned.npy')
#w2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/klm2_weight_tuned.npy')
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
weights1 = np.load(weightsdir+'/klm1_weight_rhos.npy')
weights2 = np.load(weightsdir+'/klm2_weight_rhos.npy')
#A1_bins = np.load(weightsdir+'/A1_bins_klm1_weight_tuned_data.npy')
#A2_bins = np.load(weightsdir+'/A2_bins_klm2_weight_tuned_data.npy')
A1_bins = np.load(weightsdir+'/A1_bins_klm1_weight_tuned.npy')
A2_bins = np.load(weightsdir+'/A2_bins_klm2_weight_tuned.npy')
bin_edges = np.array([0, 50, 100, 500, 1000, 1500, 2001])
# Expand to full L weights
rhos1_L = apply_bin_scalings_to_weights(weights1, bin_edges, A1_bins, lmax)
rhos2_L = apply_bin_scalings_to_weights(weights2, bin_edges, A2_bins, lmax)
rhos1_L = np.maximum(0.0, rhos1_L)
rhos2_L = np.maximum(0.0, rhos2_L)
denominator = rhos1_L + rhos2_L; denominator[denominator == 0] = 1.0 # avoid divide by zero
w1 = rhos1_L / denominator
w2 = rhos2_L / denominator
plt.figure(0)
plt.clf()
plt.plot(ell, weights1, color='firebrick', label='rhos1')
plt.plot(ell, weights2, color='darkblue', label='rhos2')
plt.plot(ell, w1, color='pink', linestyle='--', label='tuned rhos1')
plt.plot(ell, w2, color='cornflowerblue', linestyle='--', label='tuned rhos2')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.ylim(1e-2,2)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')
