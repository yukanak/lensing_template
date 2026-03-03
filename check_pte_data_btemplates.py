import numpy as np
import sys, os
import matplotlib.pyplot as plt
import btemplate as bt
import healpy as hp
from scipy.stats import chi2

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
didx = 0

a_standard, _, _ = btmp_standard.get_masked_spec(didx)
auto_standard = np.array([a_standard[digitized == i].mean() for i in range(1, len(lbins))])
a_prfhrd, _, _ = btmp_prfhrd.get_masked_spec(didx)
auto_prfhrd = np.array([a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))])
a_pp, _, _ = btmp_pp.get_masked_spec(didx)
auto_pp = np.array([a_pp[digitized == i].mean() for i in range(1, len(lbins))])

# Agora sims
idxs = np.arange(10)+5001
autos_standard_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_prfhrd_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_pp_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_prfhrd_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp_prfhrd_agora = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, _, _ = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, _, _ = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, _, _ = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]

    autos_standard_agora[ii,:] = np.array(a_standard)
    autos_prfhrd_agora[ii,:] = np.array(a_prfhrd)
    autos_pp_agora[ii,:] = np.array(a_pp)
    diff_auto_prfhrd_agora[ii,:] = np.array(a_standard) - np.array(a_prfhrd)
    diff_auto_pp_agora[ii,:] = np.array(a_standard) - np.array(a_pp)
    diff_auto_pp_prfhrd_agora[ii,:] = np.array(a_prfhrd) - np.array(a_pp)
auto_standard_agora = np.mean(autos_standard_agora, axis=0)
auto_prfhrd_agora = np.mean(autos_prfhrd_agora, axis=0)
auto_pp_agora = np.mean(autos_pp_agora, axis=0)
auto_mean_prfhrd_agora = np.mean(diff_auto_prfhrd_agora, axis=0) # sim mean to subtract
auto_mean_pp_agora = np.mean(diff_auto_pp_agora, axis=0) # sim mean to subtract
auto_mean_pp_prfhrd_agora = np.mean(diff_auto_pp_prfhrd_agora, axis=0) # sim mean to subtract

# Get error bars from sims...
idxs = np.arange(499)+1
autos_standard = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
autos_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
diff_auto_pp_prfhrd = np.zeros((len(idxs),len(lbins)-1),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]

    autos_standard[ii,:] = np.array(a_standard)
    autos_prfhrd[ii,:] = np.array(a_prfhrd)
    autos_pp[ii,:] = np.array(a_pp)
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
autos_mean_prfhrd = np.mean(autos_prfhrd, axis=0)
autos_mean_pp = np.mean(autos_pp, axis=0)

# Get the chi-squared
diff_vec_standard_prfhrd_agora = auto_mean_prfhrd_agora.real - auto_mean_prfhrd.real
diff_cov_standard_prfhrd_agora = np.cov(diff_auto_prfhrd.real, rowvar=False) / 10
diff_vec_standard_pp_agora = auto_mean_pp_agora.real - auto_mean_pp.real
diff_cov_standard_pp_agora = np.cov(diff_auto_pp.real, rowvar=False) / 10
diff_vec_prfhrd_pp_agora = auto_mean_pp_prfhrd_agora.real - auto_mean_pp_prfhrd.real
diff_cov_prfhrd_pp_agora = np.cov(diff_auto_pp_prfhrd.real, rowvar=False) / 10
diff_vec_standard_prfhrd = (np.array(auto_standard) - np.array(auto_prfhrd)) - auto_mean_prfhrd.real
diff_cov_standard_prfhrd = np.cov(diff_auto_prfhrd.real, rowvar=False)
diff_vec_standard_pp = (np.array(auto_standard) - np.array(auto_pp)) - auto_mean_pp.real
diff_cov_standard_pp = np.cov(diff_auto_pp.real, rowvar=False)
diff_vec_prfhrd_pp = (np.array(auto_prfhrd) - np.array(auto_pp)) - auto_mean_pp_prfhrd.real
diff_cov_prfhrd_pp = np.cov(diff_auto_pp_prfhrd.real, rowvar=False)
inv_diff_cov_standard_prfhrd_agora = np.linalg.inv(diff_cov_standard_prfhrd_agora)
inv_diff_cov_standard_pp_agora = np.linalg.inv(diff_cov_standard_pp_agora)
inv_diff_cov_prfhrd_pp_agora = np.linalg.inv(diff_cov_prfhrd_pp_agora)
inv_diff_cov_standard_prfhrd = np.linalg.inv(diff_cov_standard_prfhrd)
inv_diff_cov_standard_pp = np.linalg.inv(diff_cov_standard_pp)
inv_diff_cov_prfhrd_pp = np.linalg.inv(diff_cov_prfhrd_pp)
chi_sq_standard_prfhrd_agora = float(np.real(diff_vec_standard_prfhrd_agora.T @ inv_diff_cov_standard_prfhrd_agora @ diff_vec_standard_prfhrd_agora))
chi_sq_standard_pp_agora = float(np.real(diff_vec_standard_pp_agora.T @ inv_diff_cov_standard_pp_agora @ diff_vec_standard_pp_agora))
chi_sq_prfhrd_pp_agora = float(np.real(diff_vec_prfhrd_pp_agora.T @ inv_diff_cov_prfhrd_pp_agora @ diff_vec_prfhrd_pp_agora))
chi_sq_standard_prfhrd = float(np.real(diff_vec_standard_prfhrd.T @ inv_diff_cov_standard_prfhrd @ diff_vec_standard_prfhrd))
chi_sq_standard_pp = float(np.real(diff_vec_standard_pp.T @ inv_diff_cov_standard_pp @ diff_vec_standard_pp))
chi_sq_prfhrd_pp = float(np.real(diff_vec_prfhrd_pp.T @ inv_diff_cov_prfhrd_pp @ diff_vec_prfhrd_pp))
print('chi-squared, Agora, standard - prfhrd: ', chi_sq_standard_prfhrd_agora)
print('chi-squared, Agora, standard - pol-only: ', chi_sq_standard_pp_agora)
print('chi-squared, Agora, prfhrd - pol-only: ', chi_sq_prfhrd_pp_agora)
print('chi-squared, data, standard - prfhrd: ', chi_sq_standard_prfhrd)
print('chi-squared, data, standard - pol-only: ', chi_sq_standard_pp)
print('chi-squared, data, prfhrd - pol-only: ', chi_sq_prfhrd_pp)
# Translate the chi-squared to a PTE by using the number of bins as the dof
dof = len(lbins)-1
pte_standard_prfhrd_agora = chi2.sf(chi_sq_standard_prfhrd_agora, dof)
pte_standard_pp_agora = chi2.sf(chi_sq_standard_pp_agora, dof)
pte_prfhrd_pp_agora = chi2.sf(chi_sq_prfhrd_pp_agora, dof)
pte_standard_prfhrd = chi2.sf(chi_sq_standard_prfhrd, dof)
pte_standard_pp = chi2.sf(chi_sq_standard_pp, dof)
pte_prfhrd_pp = chi2.sf(chi_sq_prfhrd_pp, dof)
print('pte, Agora, standard - prfhrd: ', pte_standard_prfhrd_agora)
print('pte, Agora, standard - pol-only: ', pte_standard_pp_agora)
print('pte, Agora, prfhrd - pol-only: ', pte_prfhrd_pp_agora)
print('pte, data, standard - prfhrd: ', pte_standard_prfhrd)
print('pte, data, standard - pol-only: ', pte_standard_pp)
print('pte, data, prfhrd - pol-only: ', pte_prfhrd_pp)
# PTE per Agora realization
idxs = np.arange(10)+5001
print('Agora: per-sim...')
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, _, _ = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, _, _ = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, _, _ = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]

    d_vec_standard_prfhrd_agora = (np.array(a_standard) - np.array(a_prfhrd)) - auto_mean_prfhrd
    d_vec_standard_pp_agora = (np.array(a_standard) - np.array(a_pp)) - auto_mean_pp
    d_vec_prfhrd_pp_agora = (np.array(a_prfhrd) - np.array(a_pp)) - auto_mean_pp_prfhrd
    c_sq_standard_prfhrd_agora = float(np.real(d_vec_standard_prfhrd_agora.T @ inv_diff_cov_standard_prfhrd @ d_vec_standard_prfhrd_agora))
    c_sq_standard_pp_agora = float(np.real(d_vec_standard_pp_agora.T @ inv_diff_cov_standard_pp @ d_vec_standard_pp_agora))
    c_sq_prfhrd_pp_agora = float(np.real(d_vec_prfhrd_pp_agora.T @ inv_diff_cov_prfhrd_pp @ d_vec_prfhrd_pp_agora))
    print(f'chi-squared, Agora, standard - prfhrd, seed {idx}: ', c_sq_standard_prfhrd_agora)
    print(f'chi-squared, Agora, standard - pol-only, seed {idx}: ', c_sq_standard_pp_agora)
    print(f'chi-squared, Agora, prfhrd - pol-only, seed {idx}: ', c_sq_prfhrd_pp_agora)
    # Translate the chi-squared to a PTE by using the number of bins as the dof
    per_sim_pte_standard_prfhrd_agora = chi2.sf(c_sq_standard_prfhrd_agora, dof)
    per_sim_pte_standard_pp_agora = chi2.sf(c_sq_standard_pp_agora, dof)
    per_sim_pte_prfhrd_pp_agora = chi2.sf(c_sq_prfhrd_pp_agora, dof)
    print(f'pte, Agora, standard - prfhrd, seed {idx}: ', per_sim_pte_standard_prfhrd_agora)
    print(f'pte, Agora, standard - pol-only, seed {idx}: ', per_sim_pte_standard_pp_agora)
    print(f'pte, Agora, prfhrd - pol-only, seed {idx}: ', per_sim_pte_prfhrd_pp_agora)

# Sim-based PTE?
idxs = np.arange(499)+1
ptes_standard_prfhrd_sims = np.zeros((len(idxs)),dtype=np.complex_)
ptes_standard_pp_sims = np.zeros((len(idxs)),dtype=np.complex_)
ptes_prfhrd_pp_sims = np.zeros((len(idxs)),dtype=np.complex_)
for ii, idx in enumerate(idxs):
    print(idx)
    a_standard, c_standard, a_in_standard = btmp_standard.get_masked_spec(idx)
    a_standard = [a_standard[digitized == i].mean() for i in range(1, len(lbins))]
    a_prfhrd, c_prfhrd, a_in_prfhrd = btmp_prfhrd.get_masked_spec(idx)
    a_prfhrd = [a_prfhrd[digitized == i].mean() for i in range(1, len(lbins))]
    a_pp, c_pp, a_in_pp = btmp_pp.get_masked_spec(idx)
    a_pp = [a_pp[digitized == i].mean() for i in range(1, len(lbins))]
    mask = np.ones((len(idxs)), dtype=bool)
    mask[ii] = False
    diff_auto_prfhrd_no_ii = diff_auto_prfhrd[mask,:]
    diff_auto_pp_no_ii = diff_auto_pp[mask,:]
    diff_auto_pp_prfhrd_no_ii = diff_auto_pp_prfhrd[mask,:]
    auto_mean_prfhrd_no_ii = np.mean(diff_auto_prfhrd_no_ii, axis=0) # sim mean to subtract
    auto_mean_pp_no_ii = np.mean(diff_auto_pp_no_ii, axis=0) # sim mean to subtract
    auto_mean_pp_prfhrd_no_ii = np.mean(diff_auto_pp_prfhrd_no_ii, axis=0) # sim mean to subtract

    diff_vec_standard_prfhrd = (np.array(a_standard) - np.array(a_prfhrd)) - auto_mean_prfhrd_no_ii
    diff_cov_standard_prfhrd = np.cov(diff_auto_prfhrd_no_ii, rowvar=False)
    diff_vec_standard_pp = (np.array(a_standard) - np.array(a_pp)) - auto_mean_pp_no_ii
    diff_cov_standard_pp = np.cov(diff_auto_pp_no_ii, rowvar=False)
    diff_vec_prfhrd_pp = (np.array(a_prfhrd) - np.array(a_pp)) - auto_mean_pp_prfhrd_no_ii
    diff_cov_prfhrd_pp = np.cov(diff_auto_pp_prfhrd_no_ii, rowvar=False)
    inv_diff_cov_standard_prfhrd = np.linalg.inv(diff_cov_standard_prfhrd)
    inv_diff_cov_standard_pp = np.linalg.inv(diff_cov_standard_pp)
    inv_diff_cov_prfhrd_pp = np.linalg.inv(diff_cov_prfhrd_pp)
    chi_sq_standard_prfhrd = float(np.real(diff_vec_standard_prfhrd.T @ inv_diff_cov_standard_prfhrd @ diff_vec_standard_prfhrd))
    chi_sq_standard_pp = float(np.real(diff_vec_standard_pp.T @ inv_diff_cov_standard_pp @ diff_vec_standard_pp))
    chi_sq_prfhrd_pp = float(np.real(diff_vec_prfhrd_pp.T @ inv_diff_cov_prfhrd_pp @ diff_vec_prfhrd_pp))
    # Translate the chi-squared to a PTE by using the number of bins as the dof
    ptes_standard_prfhrd_sims[ii] = chi2.sf(chi_sq_standard_prfhrd, dof)
    ptes_standard_pp_sims[ii] = chi2.sf(chi_sq_standard_pp, dof)
    ptes_prfhrd_pp_sims[ii] = chi2.sf(chi_sq_prfhrd_pp, dof)

# Histogram
plt.clf()
plt.hist(ptes_standard_prfhrd_sims, bins=20, range=(0,np.max(ptes_standard_prfhrd_sims)), density=True, alpha=0.7)
plt.axvline(pte_standard_prfhrd, color='r', linestyle='--', linewidth=2, label='data')
plt.axvline(pte_standard_prfhrd_agora, color='b', linestyle='--', linewidth=2, label='Agora')
plt.xlabel("PTE")
plt.title("distribution of simulation PTEs, diff standard & prfhrd")
plt.xlim(0,1)
plt.legend()
plt.savefig('figs/temp.png',bbox_inches='tight')

# Check the correlation matrix of the difference covariance matrix and see if there’s significant off-diag correlation; if so, you might want to increase the binsize
#standard_dev = np.sqrt(np.diag(diff_cov.real))
#diff_corr = (diff_cov.real) / np.outer(standard_dev, standard_dev)
#plt.clf()
#plt.imshow(diff_corr, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
#plt.colorbar(label="correlation")
#plt.title("correlation matrix of diff bins")
#plt.savefig('figs/btemplates_data_spec_diff_corr_matrix.png',bbox_inches='tight')



