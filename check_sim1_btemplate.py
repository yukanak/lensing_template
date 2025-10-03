import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt

yaml_file = 'bt_gmv3500_pp.yaml'
idx = 2
btmp = bt.btemplate(yaml_file)
lmax = 4096
ell = np.arange(lmax+1)
auto, cross, auto_in = btmp.get_masked_spec(idx)
auto1, cross1, auto_in1 = btmp.get_masked_spec(1)

# Plot
plt.figure(0)
plt.clf()
plt.plot(ell, auto[:lmax+1], color='firebrick', linestyle='-', label=f'btemplate auto, sim {idx}')
plt.plot(ell, cross[:lmax+1], color='darkblue', linestyle='-', label=f'btemplate x input b cross, sim {idx}')
plt.plot(ell, auto_in[:lmax+1], color='forestgreen', linestyle='-', label=f'input b auto, sim {idx}')
plt.plot(ell, auto1[:lmax+1], color='pink', linestyle='--', label='btemplate auto, sim 1')
plt.plot(ell, cross1[:lmax+1], color='powderblue', linestyle='--', label='btemplate x input b cross, sim 1')
plt.plot(ell, auto_in1[:lmax+1], color='lightgreen', linestyle='--', label='input b auto, sim 1')
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
plt.plot(ell, (auto1/auto_in1)[:lmax+1], color='pink', linestyle='--', alpha=0.8, label='btemplate auto / input b auto, sim 1')
plt.plot(ell, (cross1/auto_in1)[:lmax+1], color='cornflowerblue', linestyle='--', alpha=0.8, label='btemplate x input b cross / input b auto, sim 1')
plt.plot(ell, (auto_in/auto_in1)[:lmax+1], color='bisque', linestyle='--', alpha=0.8, label=f'input b auto, sim {idx} / input b auto, sim 1')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0,1.3)
plt.legend(loc='upper right', fontsize='x-small')
plt.title(f'btemplate check')
plt.xlabel('$\ell$')
plt.savefig('figs/btemplate_check_ratio.png',bbox_inches='tight')
