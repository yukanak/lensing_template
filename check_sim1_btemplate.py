import numpy as np
import healpy as hp
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

idx = 1
#yaml_file = 'bt_gmv3500_combined_lenz.yaml'
yaml_file = 'bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml'
btmp = bt.btemplate(yaml_file, combined_tracer=True)
lmax = 2000
ell = np.arange(lmax+1)
#auto, cross, auto_in = btmp.get_masked_spec(idx)
#auto1, cross1, auto_in1 = btmp.get_masked_spec(1)
fname = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_pr3_cib_pr4_kappa_standard/btmpl_specs_0001.npz'
tmp   = np.load(fname)
auto  = tmp['auto']
cross = tmp['cross']
auto_in  = tmp['auto_in']
#fname = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/combined_cib_qe_pr3_cib_pr4_kappa_standard/tuned/btmpl_specs_0001.npz'
#tmp   = np.load(fname)
#auto1  = tmp['auto']
#cross1 = tmp['cross']
#auto_in1  = tmp['auto_in']
#w1 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm1_weight_tuned_seed1.npy')
#w2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm2_weight_tuned_seed1.npy')
#klm1 = btmp.get_debiased_klm(1)
#klm1 = utils.reduce_lmax(klm1, lmax=lmax)
#klm2_map = hp.read_map(btmp.cib_tracer_dir + f'/cib_tracer_seed1.fits')
#klm2 = hp.map2alm(klm2_map, lmax=lmax)
#klm_combined = hp.almxfl(klm1,w1) + hp.almxfl(klm2,w2)
#klm_combined = np.nan_to_num(klm_combined, nan=0.0, posinf=0.0, neginf=0.0)
#hp.write_alm(btmp.dir_combined_tracer+f'/klm_combined_cib_qe_seed1_tuned.alm', klm_combined, overwrite=True)
#auto1, cross1, auto_in1 = btmp.get_masked_spec(1,overwrite=True,savefile=True)
auto1, cross1, auto_in1 = btmp.get_masked_spec(1,overwrite=False)

# Plot
plt.figure(0)
plt.clf()
plt.plot(ell, auto[:lmax+1], color='firebrick', linestyle='-', label=f'btemplate auto, sim {idx}')
plt.plot(ell, cross[:lmax+1], color='darkblue', linestyle='-', label=f'btemplate x input b cross, sim {idx}')
plt.plot(ell, auto_in[:lmax+1], color='forestgreen', linestyle='-', label=f'input b auto, sim {idx}')
plt.plot(ell, auto1[:lmax+1], color='pink', linestyle='--', label='btemplate auto, sim 1, TUNED')
plt.plot(ell, cross1[:lmax+1], color='powderblue', linestyle='--', label='btemplate x input b cross, sim 1, TUNED')
plt.plot(ell, auto_in1[:lmax+1], color='lightgreen', linestyle='--', label='input b auto, sim 1, TUNED')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-7,3e-6)
plt.legend(loc='lower left', fontsize='x-small')
plt.title(f'btemplate check')
plt.ylabel("$C_\ell^{BB}$")
plt.xlabel('$\ell$')
plt.savefig('figs/btemplate_check_spec.png',bbox_inches='tight')

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
plt.plot(ell, (auto/auto_in)[:lmax+1], color='firebrick', linestyle='-', alpha=0.8, label=f'btemplate auto / input b auto, sim {idx}')
plt.plot(ell, (cross/auto_in)[:lmax+1], color='darkblue', linestyle='-', alpha=0.8, label=f'btemplate x input b cross / input b auto, sim {idx}')
plt.plot(ell, (auto1/auto_in1)[:lmax+1], color='pink', linestyle='--', alpha=0.8, label='btemplate auto / input b auto, sim 1, TUNED')
plt.plot(ell, (cross1/auto_in1)[:lmax+1], color='cornflowerblue', linestyle='--', alpha=0.8, label='btemplate x input b cross / input b auto, sim 1, TUNED')
plt.plot(ell, (auto/auto1)[:lmax+1], color='lightgreen', linestyle='--', alpha=0.8, label=f'btemplate auto, sim 1 NOT TUNED / TUNED')
plt.plot(ell, (cross/cross1)[:lmax+1], color='orange', linestyle='--', alpha=0.8, label=f'btemplate x input b cross, sim 1 NOT TUNED / TUNED')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0,1.3)
plt.legend(loc='upper right', fontsize='x-small')
plt.title(f'btemplate check')
plt.xlabel('$\ell$')
plt.savefig('figs/btemplate_check_ratio.png',bbox_inches='tight')

plt.figure(0)
plt.clf()
weightsdir = btmp.combined_tracer_weights_dir
weights1 = np.load(weightsdir+'/klm1_weight.npy')
weights2 = np.load(weightsdir+'/klm2_weight.npy')
w1 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm1_weight_tuned_seed1.npy')
w2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/combined_tracer_weights/combined_qe_pr3_cib_pr4_kappa_tracer/checkpoint_klm2_weight_tuned_seed1.npy')
plt.plot(ell, weights1, color='firebrick', label='w1')
plt.plot(ell, weights2, color='darkblue', label='w2')
plt.plot(ell, w1, color='pink', linestyle='--', label='tuned w1')
plt.plot(ell, w2, color='cornflowerblue', linestyle='--', label='tuned w2')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$\ell$')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,2000)
plt.tight_layout()
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')
