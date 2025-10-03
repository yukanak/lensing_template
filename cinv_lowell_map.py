#!/usr/bin/env python
import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/users/yukanaka/spt3g_software/scratch/wlwu/cinv_lowell/')
sys.path.insert(0, '/home/users/yukanaka/healqest/healqest/src/')
sys.path.insert(0, '/home/users/yukanaka/healqest/pipeline/')
import maps as hq_maps
import healqest_utils as utils
from cinv import cinv_hp as cinv
from cinv import cinv_hp_yuka as cinv_old
import cinv_lowell as cll

# See /home/users/yukanaka/spt3g_software/scratch/wlwu/cinv_lowell/script24_0830_ivf_invvar.py
idx = int(sys.argv[1])
params = yaml.safe_load(open("/home/users/yukanaka/lensing_template/lowell_v3mocks_musebeamv41.yaml"))

dir_tmp = params['outdir']
outdir = dir_tmp+'cinv_output_test_uncorr/'

nlev_p_uk = 5/np.sqrt(2) # at ell < 500, more like 10 uK-arcmin
nvar      = (nlev_p_uk*np.pi/180./60.)**2 # ~ 1e-6 uK^2-steradian
lmax      = params['lmax']
lmin      = 20
nside     = 2048

mask  = cll.get_mask(params)
fsky  = cll.get_fsky()

nl2d_ee = cll.get_smooth_nl2d(params, eorb='ee')/fsky
nl2d_bb = cll.get_smooth_nl2d(params, eorb='bb')/fsky
nl2d_ee -= nvar
nl2d_bb -= nvar
nl2d_ee[nl2d_ee < 0] = 1e-12
nl2d_bb[nl2d_bb < 0] = 1e-12
nl2d_ee   = nl2d_ee.astype(np.complex_)
nl2d_bb   = nl2d_bb.astype(np.complex_)
dict_nl2d = {'tt':nl2d_ee, 'ee':nl2d_ee, 'bb':nl2d_bb}

# N^-1; inverse pixel variance in 1/uK^2 
ninv_p = mask *( 1/nvar *hp.nside2pixarea(nside) )

# tf x beam: 2d and 1d (inv nvar weighted)
tf2d = cll.get_inv_nvar_weighted_2dtfbeam(params, freqs=[90,150])
tf1d = cll.get_inv_nvar_weighted_1dtfbeam(params, freqs=[90,150])

# Cls
clfname  = "/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat"
cl_len   = utils.get_lensedcls(clfname,lmax=lmax,dict=True)

sim_dict = {}
sim_dict['file_signal'] = params['outdir']+params['maps']['invnvar_map_fname'] # inv nvar weighted signal+noise 
sim_dict['nside']     = nside
sim_dict['file_mask'] = mask
sim_lib     = hq_maps.maps(sim_dict)

eps_min = 0.001
'''
print('Getting the old ver!')
cinv_p_old = cinv_old.cinv_p(outdir+"/P/",
                     lmax,
                     nside,
                     cl_len, tf1d, [ninv_p],
                     eps_min = eps_min,
                     nl      = dict_nl2d,
                     tf2d    = tf2d)
cinv_t_old = cinv_old.cinv_t(outdir+"/T/",
                     lmax,
                     nside,
                     cl_len, tf1d, [ninv_p],
                     eps_min = eps_min,
                     nl      = dict_nl2d,
                     tf2d    = None)
'''
print('Getting the new ver!')
cinv_p = cinv.cinv_p(outdir+"/P/",
                     lmax,
                     nside,
                     cl_len,
                     dict_nl2d,
                     [ninv_p],
                     tf1d, tf1d,
                     tf2d, tf2d,
                     eps_min = eps_min,
                     )
cinv_t = cinv.cinv_t(outdir+"/T/",
                     lmax,
                     nside,
                     cl_len,
                     dict_nl2d,
                     [ninv_p],
                     tf1d,
                     tf2d = None,
                     eps_min = eps_min,
                     )

lfilt    = np.ones(lmax + 1, dtype=float) * (np.arange(lmax + 1) >= lmin)
# put cinv_p in cinv_t slot as placeholder; DO NOT RUN T cinv!
#ivfs     = cinv.library_cinv_sepTP(dir_tmp, sim_lib, cinv_t, cinv_p, cl_len, lfilt = lfilt) # old beam inv nvar maps
'''
print('Getting the old ver ivfs!')
ivfs_old     = cinv_old.library_cinv_sepTP(outdir, sim_lib, cinv_t_old, cinv_p_old, cl_len, lfilt = lfilt)
'''
print('Getting the new ver ivfs!')
ivfs     = cinv.library_cinv_sepTP(outdir, sim_lib, cinv_t, cinv_p, cl_len, lfilt = lfilt)

#for idx in np.arange(500)+1:
print('Getting the new ver ivfs get_sim_emliklm!')
ivf_e    = ivfs.get_sim_emliklm(idx)
'''
print('Getting the old ver ivfs get_sim_emliklm!')
ivf_e_old    = ivfs_old.get_sim_emliklm(idx)
'''

print('Output saved')

# output saved at dir_tmp as
# /lcrc/project/SPT3G/analysis/cinv_lowellmap_19-20/v3mocks/
# sim_%04i_elm.fits

# plot check
if 0:
    eb = "ee"

    nl2d_ee_o = np.load(dir_tmp + params['noise']['nl2d_smooth']['fname']%eb)/fsky
    nl1d_ee = hp.alm2cl(np.sqrt(nl2d_ee_o.astype(np.complex_)))
    wf = tf1d**2 * cl_len[eb] / (tf1d**2 * cl_len[eb] + nl1d_ee)

    plt.figure()
    i = 1
    elm_inv = hp.read_alm(outdir+"sim_000%i_%slm.fits"%(i,eb[0]))
    plt.plot(hp.alm2cl(hp.almxfl(elm_inv, cl_len[eb]))/fsky, lw=0.5)
    plt.plot(cl_len[eb]*tf1d**2,':',color="gray", label="$C_{\ell}^{%s}$ * TF^2"%eb.upper())
    plt.plot(nl1d_ee,'--',color="gray", label="Nl" )
    plt.plot(wf*cl_len[eb], 'k--', label="WF * $C_{\ell}^{%s}$"%eb.upper())

