import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

idx = 1

lmax = 1500
nside = 2048
l = np.arange(lmax+1)
dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cib_phi_tracer/'
# there are 8 cib['cibxkap'] and cib['cibauto'] spectra in the file
# from the 8 high Galactic latitude patches as noted in the BKSPT delensing paper
cib = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/patches_cibxkap_cibauto_clkk.npz")
#rho = cib['cibxkap']/np.sqrt(cib['cibauto']*cib['clkk'])
# to start, you can take the mean of them as the cib x kappa and cib-auto values
clik = np.nanmean(cib['cibxkap'], axis=0)
clii = np.nanmean(cib['cibauto'], axis=0)
clkk = cib['clkk']

# MAKE CIB SIMS FROM INPUT PHI (https://arxiv.org/pdf/2011.08163)
input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{idx}_lmax4096.alm')
ell, m = hp.Alm.getlm(4096)
fac = np.zeros_like(ell, dtype=float)
fac[ell>=2] = -0.5 * ell[ell>=2]*(ell[ell>=2]+1.0)
input_klm = hp.almxfl(input_plm, fac)
# signal
I_sig = hp.almxfl(input_klm, clik/clkk)
# noise
nl = clii - clik**2/clkk
I_noi = hp.synalm(nl, lmax=lmax, new=True)
# total
I_alm = I_sig + I_noi
I_map = hp.alm2map(I_alm, nside=nside, lmax=lmax)

# MAKE CIB-BASED PHI TRACER
klm = hp.almxfl(I_alm, clik/clii)

# the spectra have a lot of wiggles, which are not real structure
# (patches are small so the minimum resolution in ell is 10s of ells)
# so smoothing before feeding them to the rest of the pipeline is good

