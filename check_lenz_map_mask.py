#!/usr/bin/env python
import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
from scipy.ndimage import gaussian_filter1d
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

mask_spt = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits')
#mask_lenz = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/2.5e+20_gp20/mask_apod.hpx.fits")
mask_lenz = hp.read_map("/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/lenz_et_al_cib/4.0e+20_gp40/mask_apod.hpx.fits")

nside = 2048
if hp.get_nside(mask_lenz) != nside:
    mask_lenz = hp.ud_grade(mask_lenz, nside, order_in="RING", order_out="RING")
if hp.get_nside(mask_spt) != nside:
    mask_spt  = hp.ud_grade(mask_spt,  nside, order_in="RING", order_out="RING")

# binarize (important if apodized)
ml = mask_lenz > 0
ms = mask_spt  > 0

npix = hp.nside2npix(nside)
fsky = 1.0 / npix   # each pixel's sky fraction

# regions
both   = ml & ms
only_l = ml & ~ms
only_s = ~ml & ms
neither= ~ml & ~ms

# sky fractions
f_both   = both.sum()   * fsky
f_lenz   = ml.sum()     * fsky
f_spt    = ms.sum()     * fsky
f_only_l = only_l.sum() * fsky
f_only_s = only_s.sum() * fsky
f_neither= neither.sum()* fsky

print("Sky fractions:")
print(f"  Lenz mask total        : {f_lenz:.4f}")
print(f"  SPT mask total         : {f_spt:.4f}")
print(f"  Overlap (Lenz ∩ SPT)   : {f_both:.4f}")
print(f"  Only Lenz              : {f_only_l:.4f}")
print(f"  Only SPT               : {f_only_s:.4f}")
print(f"  Neither                : {f_neither:.4f}")

print("\nConditional coverages:")
print(f"  Fraction of SPT inside Lenz : {f_both / f_spt:.3f}")
print(f"  Fraction of Lenz inside SPT : {f_both / f_lenz:.3f}")

hp.mollview(ml.astype(int) + 2*ms.astype(int),
            title="Mask overlap: RED=Lenz, PINK=SPT, TEAL=Overlap",
            cmap="tab10")
plt.savefig('/home/users/yukanaka/lensing_template/figs/temp.png')

