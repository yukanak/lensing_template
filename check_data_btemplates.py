import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp

yaml_file = 'bt_gmv3500.yaml'
yaml_file_prfhrd = 'bt_gmv3500_prfhrd.yaml'
yaml_file_pp = 'bt_gmv3500_pp.yaml'
btmp_standard = bt.btemplate(yaml_file)
btmp_prfhrd = bt.btemplate(yaml_file_prfhrd)
btmp_pp = bt.btemplate(yaml_file_pp)
lmax = 4096
l = np.arange(lmax+1)
lbins = np.logspace(np.log10(30),np.log10(1000),20)
#lbins = np.logspace(np.log10(30),np.log10(1000),10)
#lbins = np.linspace(30, 1000, 21)
#lbins = np.linspace(30, 1000, 11)
#lbins = np.linspace(30, 300, 11)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(np.arange(6144), lbins)
nside = 2048
mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense.fits')
scal = 1e8
fwhm_arcmin = 20.0
fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # convert arcmin to radians
bl = hp.gauss_beam(fwhm=fwhm_rad, lmax=4096)

bt_standard_sim1 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500_test_uncorr/btmpl_alm_0001.fits')
bt_standard_sim1 = hp.almxfl(bt_standard_sim1, bl)
bt_standard_sim1_map = hp.alm2map(bt_standard_sim1,nside=nside)
bt_prfhrd_sim1 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500_test_uncorr_prfhrd/btmpl_alm_0001.fits')
bt_prfhrd_sim1 = hp.almxfl(bt_prfhrd_sim1, bl)
bt_prfhrd_sim1_map = hp.alm2map(bt_prfhrd_sim1,nside=nside)
bt_pp_sim1 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/sqe_lmaxT3500_test_uncorr_pp/btmpl_alm_0001.fits')
bt_pp_sim1 = hp.almxfl(bt_pp_sim1, bl)
bt_pp_sim1_map = hp.alm2map(bt_pp_sim1,nside=nside)
diff_sim1 = bt_standard_sim1_map - bt_prfhrd_sim1_map
diff_sim1_pp = bt_standard_sim1_map - bt_pp_sim1_map

bt_standard = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500_test_uncorr/btmpl_alm_0000.fits')
bt_standard = hp.almxfl(bt_standard, bl)
bt_standard_map = hp.alm2map(bt_standard,nside=nside)
bt_prfhrd = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/gmvjtp_sep_lmaxT3500_test_uncorr_prfhrd/btmpl_alm_0000.fits')
bt_prfhrd = hp.almxfl(bt_prfhrd, bl)
bt_prfhrd_map = hp.alm2map(bt_prfhrd,nside=nside)
bt_pp = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/lensing_template/btemplates/sqe_lmaxT3500_test_uncorr_pp/btmpl_alm_0000.fits')
bt_pp = hp.almxfl(bt_pp, bl)
bt_pp_map = hp.alm2map(bt_pp,nside=nside)
diff = bt_standard_map - bt_prfhrd_map
diff_pp = bt_standard_map - bt_pp_map
diff_pp_prfhrd = bt_prfhrd_map - bt_pp_map

a_standard, _, _ = btmp_standard.get_masked_spec(0)
auto_standard = np.array([a_standard[digitized == i].mean() for i in range(1, len(lbins))])
a_prfhrd, _, _ = btmp_prfhrd.get_masked_spec(0)
auto_prfhrd = np.array([a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))])
a_pp, _, _ = btmp_pp.get_masked_spec(0)
auto_pp = np.array([a_pp[digitized == i].mean() for i in range(1, len(lbins))])

# Get error bars from sims...
idxs = np.arange(499)+1
autos_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
cross_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_in_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_in_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_in_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    c_standard = [c_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_in_standard = [a_in_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    c_prfhrd = [c_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_in_prfhrd = [a_in_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]
    c_pp = [c_pp[digitized == i].mean() for i in range(1, len(lbins))]
    a_in_pp = [a_in_pp[digitized == i].mean() for i in range(1, len(lbins))]

    autos_standard[ii,:] = np.array(a_standard)
    autos_prfhrd[ii,:] = np.array(a_prfhrd)
    autos_pp[ii,:] = np.array(a_pp)
    cross_standard[ii,:] = np.array(c_standard)
    cross_prfhrd[ii,:] = np.array(c_prfhrd)
    cross_pp[ii,:] = np.array(c_pp)
    autos_in_standard[ii,:] = np.array(a_in_standard)
    autos_in_prfhrd[ii,:] = np.array(a_in_prfhrd)
    autos_in_pp[ii,:] = np.array(a_in_pp)
    diff_auto_prfhrd[ii,:] = np.array(a_standard) - np.array(a_prfhrd)
    diff_auto_pp[ii,:] = np.array(a_standard) - np.array(a_pp)
    diff_auto_pp_prfhrd[ii,:] = np.array(a_prfhrd) - np.array(a_pp)
auto_std_prfhrd = np.std(diff_auto_prfhrd, axis=0)
auto_std_pp = np.std(diff_auto_pp, axis=0)
auto_std_pp_prfhrd = np.std(diff_auto_pp_prfhrd, axis=0)
auto_mean_prfhrd = np.mean(diff_auto_prfhrd, axis=0) # sim mean to subtract
auto_mean_pp = np.mean(diff_auto_pp, axis=0) # sim mean to subtract
auto_mean_pp_prfhrd = np.mean(diff_auto_pp_prfhrd, axis=0) # sim mean to subtract
autos_mean_standard = np.mean(autos_standard, axis=0)
cross_mean_standard = np.mean(cross_standard, axis=0)
autos_input_mean_standard = np.mean(autos_in_standard, axis=0)
autos_mean_prfhrd = np.mean(autos_prfhrd, axis=0)
cross_mean_prfhrd = np.mean(cross_prfhrd, axis=0)
autos_input_mean_prfhrd = np.mean(autos_in_prfhrd, axis=0)
autos_mean_pp = np.mean(autos_pp, axis=0)
cross_mean_pp = np.mean(cross_pp, axis=0)
autos_input_mean_pp = np.mean(autos_in_pp, axis=0)
r_std = cross_mean_standard / np.sqrt(autos_mean_standard * autos_input_mean_standard)
r_prfhrd = cross_mean_prfhrd / np.sqrt(autos_mean_prfhrd * autos_input_mean_prfhrd)
r_pp = cross_mean_pp / np.sqrt(autos_mean_pp * autos_input_mean_pp)

# Plot
plt.figure(0)
plt.clf()
plt.plot(np.zeros(4096),'k--',lw=0.8,dashes=(6,3))
plt.errorbar(bin_centers, ((np.array(auto_standard) - np.array(auto_prfhrd)) - auto_mean_prfhrd) * scal, yerr=auto_std_prfhrd * scal, color='firebrick', linestyle='-', label="btemplate auto diff standard - prfhrd")
plt.errorbar(bin_centers, ((np.array(auto_standard) - np.array(auto_pp)) - auto_mean_pp) * scal, yerr=auto_std_pp * scal, color='darkblue', linestyle='-', label="btemplate auto diff standard - pol-only")
plt.errorbar(bin_centers, ((np.array(auto_prfhrd) - np.array(auto_pp)) - auto_mean_pp_prfhrd) * scal, yerr=auto_std_pp_prfhrd * scal, color='forestgreen', linestyle='-', label="btemplate auto diff prfhrd - pol-only")
plt.legend(loc='upper right', fontsize='medium')
plt.xscale('log')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\Delta C_\ell^{BB} / 10^{-8}$")
plt.xlim(10,2000)
plt.savefig('figs/btemplates_data_spec_diff.png',bbox_inches='tight')

# Diff vs correlation coefficient
plt.figure(0)
plt.clf()
plt.scatter(((np.array(auto_standard) - np.array(auto_prfhrd)) - auto_mean_prfhrd) * scal, r_prfhrd, color='firebrick', linestyle='-', label="prfhrd", s=10)
plt.scatter(((np.array(auto_standard) - np.array(auto_pp)) - auto_mean_pp) * scal, r_pp, color='darkblue', linestyle='-', label="pol-only", s=10)
plt.scatter(np.zeros_like(auto_standard), r_std, color="gray", label="standard", s=10)
plt.axhline(0, color="darkgray", lw=1, linestyle="--")
plt.axvline(0, color="darkgray", lw=1, linestyle="--")
plt.xlabel(r"btemplate auto diff (standard - alt tracer)")
plt.ylabel(r"correlation coefficient $r$")
plt.title(r"correlation coefficient vs $\Delta C_\ell^{BB}$ across bins")
plt.legend()
plt.ylim(0.4,0.8)
plt.savefig('figs/btemplates_correlation_coeff_vs_data_spec_diff.png',bbox_inches='tight')

# Get the chi-squared
#diff_vec = (np.array(auto_standard) - np.array(auto_prfhrd)) - auto_mean_prfhrd
#diff_cov = np.cov(diff_auto_prfhrd, rowvar=False)
#diff_vec = (np.array(auto_standard) - np.array(auto_pp)) - auto_mean_pp
#diff_cov = np.cov(diff_auto_pp, rowvar=False)
diff_vec = (np.array(auto_prfhrd) - np.array(auto_pp)) - auto_mean_pp_prfhrd
diff_cov = np.cov(diff_auto_pp_prfhrd, rowvar=False)
inv_diff_cov = np.linalg.inv(diff_cov)
chi_sq = float(np.real(diff_vec.T @ inv_diff_cov @ diff_vec))
print('chi-squared: ', chi_sq)
# Translate the chi-squared to a PTE by using the number of bins as the dof
from scipy.stats import chi2
dof = len(lbins)-1
pte = chi2.sf(chi_sq, dof)
print('pte: ', pte)
# Check the correlation matrix of the difference covariance matrix and see if there’s significant off-diag correlation; if so, you might want to increase the binsize
standard_dev = np.sqrt(np.diag(diff_cov.real))
diff_corr = (diff_cov.real) / np.outer(standard_dev, standard_dev)
plt.clf()
plt.imshow(diff_corr, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="correlation")
plt.title("correlation matrix of diff bins")
plt.savefig('figs/btemplates_data_spec_diff_corr_matrix.png',bbox_inches='tight')

##### ABHI'S CODE #####
plt.clf()

# Display map with apodization visible
disp = diff_pp_prfhrd * mask
#disp = diff_sim1_pp * mask
#disp = diff_pp * mask
#disp = diff_sim1 * mask
#disp = diff * mask
#disp = bt_standard_map * mask
disp[mask == 0] = hp.UNSEEN   # only zero-mask outside is hidden

#rms = np.std(diff[mask > 0])
#vmin, vmax = -2*rms, 2*rms
vmin, vmax = -0.2, 0.2
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
plt.text(0.45, 0.07, "Right Ascension", fontsize=14,
         ha='center', va='bottom', transform=plt.gcf().transFigure)
plt.text(0.017, 0.25, "Declination", fontsize=14,
         ha='left', rotation=141, # 'vertical'
         transform=plt.gcf().transFigure)

# Colorbar
ax = plt.gca()
im = ax.get_images()[0]
cbar = plt.colorbar(im, orientation='vertical', fraction=0.05, pad=0.07)
#cbar.set_label(r'$y \times 10^6$', fontsize=14)

plt.title(r"Lensing Template Difference Profile Hardened - Pol-Only", fontsize=14)
#plt.title(r"Lensing Template Difference Standard - Pol-Only, Sim 1", fontsize=14)
#plt.title(r"Lensing Template Difference Standard - Pol-Only", fontsize=14)
#plt.title(r"Lensing Template Difference Standard - Profile Hardened, Sim 1", fontsize=14)
#plt.title(r"Lensing Template Difference Standard - Profile Hardened", fontsize=14)
#plt.title(r"Lensing Template Standard", fontsize=14)

# Tight layout and save
plt.tight_layout()
plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_diff_pp_prfhrd.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_diff_pp_sim1.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_diff_pp.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_diff_sim1.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_diff.png", dpi=300, bbox_inches='tight')  # Save high-res
#plt.savefig("/home/users/yukanaka/lensing_template/figs/btemplate_data_map_standard.png", dpi=300, bbox_inches='tight')  # Save high-res




