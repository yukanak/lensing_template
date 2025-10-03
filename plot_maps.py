import numpy as np
import pickle
import healpy as hp
import camb
import matplotlib.pyplot as plt
import os, sys

def alm_cutlmax(almin,new_lmax):
    '''
    Get a new alm with a smaller lmax.
    Note that in an alm array, values where m > l are left out, because they are zero.
    '''
    # getidx takes args (old) lmax, l, m and returns an array of indices for new alm
    lmmap = hp.Alm.getidx(hp.Alm.getlmax(np.shape(almin)[-1]),
                          *hp.Alm.getlm(new_lmax,np.arange(hp.Alm.getsize(new_lmax))))
    nreal = np.shape(almin)[0]

    if nreal <= 3:
        # Case if almin is a list of T, E and B alms and not just a single alm
        almout = np.zeros((nreal,hp.Alm.getsize(new_lmax)),dtype=np.complex_)
        for i in range(nreal):
            almout[i] = almin[i][lmmap]
    else:
        almout = np.zeros(hp.Alm.getsize(new_lmax),dtype=np.complex_)
        almout = almin[lmmap]

    return almout

nside = 2048
mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits')
fwhm_arcmin = 20.0 #5.0
fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # convert arcmin to radians
fwhm_arcmin2 = 30.0 #5.0
fwhm_rad2 = np.radians(fwhm_arcmin2 / 60.0)  # convert arcmin to radians

cinv_elm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/cinv_lowellmap_19-20/v3mocks_musebeamv41_uncorr/cinv_output_test_uncorr/sim_0001_elm.fits')
bl = hp.gauss_beam(fwhm=fwhm_rad, lmax=hp.Alm.getlmax(len(cinv_elm)))
cinv_elm = hp.almxfl(cinv_elm, bl) # smooth with 5' beam
#cinv_elm = alm_cutlmax(cinv_elm,300)
cinv_emap = hp.alm2map(cinv_elm,nside=nside)

kmap = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/kmapxx_kGMV_1_1.fits')
kmap = hp.smoothing(kmap, fwhm=fwhm_rad)

btemplate = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500_test_uncorr/btmpl_alm_0001.fits')
bl = hp.gauss_beam(fwhm=fwhm_rad2, lmax=hp.Alm.getlmax(len(btemplate)))
btemplate = hp.almxfl(btemplate, bl) # smooth
#btemplate = alm_cutlmax(btemplate,300)
btemplate_map = hp.alm2map(btemplate,nside=nside)

input_b = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/lmax5000/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_lmax17000_nside8192_interp1.6_method1_pol_1_alms_lowpass5000.fits', hdu=[3])
bl = hp.gauss_beam(fwhm=fwhm_rad2, lmax=5000)
input_b = hp.almxfl(input_b, bl) # smooth
#input_b = alm_cutlmax(input_b,300)
input_b_map = hp.alm2map(input_b, nside)

##### ABHI'S CODE #####
plt.clf()

# Display map with apodization visible
#disp = cinv_emap * mask
#disp = kmap * mask
disp = btemplate_map * mask
#disp = input_b_map * mask

disp[mask == 0] = hp.UNSEEN   # only zero-mask outside is hidden

#vmin, vmax = -15, 20 # Abhi's original
#vmin, vmax = -3e4, 3e4 # E
#vmin, vmax = -0.1, 0.1 # kappa
vmin, vmax = -0.5, 0.5 # B
cmap = plt.cm.coolwarm

# plt.figure(figsize=(10, 10))
hp.azeqview(
    disp,
    rot=(0, -59, 0.0),
    half_sky=True,
    xsize=4500, ysize=2300,
    reso=1.0,
    cmap=cmap,
    min=vmin, max=vmax,
    badcolor='white',
    notext=True,
    cbar=False,
)

"""
# Graticule
hp.graticule(dpar=10, dmer=30, coord='C',
             alpha=0.6, color='0.35', linestyle='--', linewidth=0.8)
"""
# Custom declination lines
for dec_deg in [-70, -60, -50, -40]:   # only these
    ra_vals = np.linspace(0, 360, 361)
    dec_vals = np.ones_like(ra_vals) * dec_deg
    hp.projplot(ra_vals, dec_vals, lonlat=True,
                color='0.35', alpha=0.6, linestyle='--', linewidth=0.8)

# Custom RA lines
for ra_deg in np.arange(0, 360, 30):
    dec_vals = np.linspace(-80, -30, 200)   # span the Dec range
    ra_vals = np.ones_like(dec_vals) * ra_deg
    hp.projplot(ra_vals, dec_vals, lonlat=True,
                color='0.35', alpha=0.6, linestyle='--', linewidth=0.8)

# RA labels (hours)
for ra_deg in [0, 30, 60, 300, 330]:#np.arange(0, 360, 30):
    hp.projtext(
        ra_deg, -76, f"{int((ra_deg/15)%24)}h",
        lonlat=True, coord='C',
        color='0.1', fontsize=13,
        ha='center', va='top'
    )

# Dec labels (degrees)
for dec_deg in np.arange(-70, -30, 10):
    hp.projtext(
        57, dec_deg, f"{int(dec_deg)}°",
        lonlat=True, coord='C',
        color='0.1', fontsize=13,
        ha='left', va='center'
    )

# overall axis-like labels using plt.text
plt.text(0.5, 0.01, "Right Ascension", fontsize=14,
         ha='center', va='bottom', transform=plt.gcf().transFigure)
#plt.text(0.45, 0.07, "Right Ascension", fontsize=14,
#         ha='center', va='bottom', transform=plt.gcf().transFigure)
plt.text(0.017, 0.25, "Declination", fontsize=14,
         ha='left', rotation=141, # 'vertical'
         transform=plt.gcf().transFigure)

# Colorbar
#ax = plt.gca()
#im = ax.get_images()[0]
#cbar = plt.colorbar(im, orientation='vertical', fraction=0.05, pad=0.07)
#cbar.set_label(r'$y \times 10^6$', fontsize=14)

#plt.title(r"E Map (Low-$\ell$ Map Set)", fontsize=14)
#plt.title(r"$\kappa$ Map from SPT-3G 2019/2020 Lensing Analysis", fontsize=14)
plt.title(r"Lensing Template", fontsize=14)
#plt.title(r"Input B Map", fontsize=14)

# Tight layout and save
plt.tight_layout()
#plt.savefig("/home/users/yukanaka/lensing_template/figs/cinv_e_map.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/kappa_map.png", dpi=300, bbox_inches='tight')  # Save high-res
plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_map.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/input_b_map.png", dpi=300, bbox_inches='tight')  # Save high-res



