import sys, os
import numpy as np
import healpy as hp
from astropy.io import fits
import yaml
sys.path.insert(0,'/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as hutils
import matplotlib.pyplot as plt

def parse_path(yaml):
    config = hutils.load_yaml(yaml)
    psname = config["pspec"]["psname"]

    dir_base = config["lensrec"]["dir_out"].format(
        rectype=config["lensrec"]["rectype"],
        runname=config["base"]["runname"],
        lmaxT=config["lensrec"]["lmaxT"],
        lminT=config["lensrec"]["lminT"],
        lminE=config["lensrec"]["lminE"],
        lminB=config["lensrec"]["lminB"],
        lmaxE=config["lensrec"]["lmaxE"],
        lmaxB=config["lensrec"]["lmaxB"],
        mmin=config["lensrec"]["mmin"],
    )

    if psname is not None and psname != "":
        dir_cls = dir_base + f"/clkk_polspice_{psname}_nops/"
    else:
        dir_cls = dir_base + f"/clkk_polspice_nops/"

    return dir_cls

n0_std = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_mh.yaml'),498,'gmv','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
n0_prfhrd = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_prfhrd.yaml'),498,'gmvbhttprf','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
n0_pp = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_sqe.yaml'),498,'qpp','N0',N0=None,Lmin=24,Lmax=2500,use_cache=True,verbose=False)
rdn0_std = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_mh.yaml'),498,'gmv','RDN0',N0=n0_std,Lmin=24,Lmax=2500,use_cache=True,verbose=False,didx=0)
rdn0_prfhrd = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_prfhrd.yaml'),498,'gmvbhttprf','RDN0',N0=n0_prfhrd,Lmin=24,Lmax=2500,use_cache=True,verbose=False,didx=0)
rdn0_pp = hutils.loadcls(parse_path('/home/users/yukanaka/healqest/pipeline/spt3g_20192020/yaml/gmv/config_gmv_052425_crosstf_lmin500_500_500_lmax3500_3000_3000_mmin100_v4_sqe.yaml'),498,'qpp','RDN0',N0=n0_pp,Lmin=24,Lmax=2500,use_cache=True,verbose=False,didx=0)
np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/sqe052425/sqe/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/RDN0_QPP.npy",rdn0_pp)
#np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/SAN0/RDN0_GMVBHTTPRF.npy",rdn0_prfhrd)
#np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/RDN0_GMV.npy",rdn0_std)

san0_pp = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/sqe052425/sqe/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_PP_spice.npy")[:,1:]
san0_prfhrd = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/SAN0/SAN0_array_GMVBHTTPRF_spice.npy")[:,1:]
san0_std = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_GMV_spice.npy")[:,1:]
san0_tf_pp = np.mean(san0_pp,axis=1)/n0_pp
san0_tf_prfhrd = np.mean(san0_prfhrd,axis=1)/n0_prfhrd
san0_tf_std = np.mean(san0_std,axis=1)/n0_std
san0_pp_nrm = san0_pp / san0_tf_pp[:,np.newaxis]
san0_prfhrd_nrm = san0_prfhrd / san0_tf_prfhrd[:,np.newaxis]
san0_std_nrm = san0_std / san0_tf_std[:,np.newaxis]
san0_pp_nrm[np.isnan(san0_pp_nrm)]=0
san0_prfhrd_nrm[np.isnan(san0_prfhrd_nrm)]=0
san0_std_nrm[np.isnan(san0_std_nrm)]=0
np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/sqe052425/sqe/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_QPP_spice_nrm.npy",san0_pp_nrm)
#np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/SAN0/SAN0_array_GMVBHTTPRF_spice_nrm.npy",san0_prfhrd_nrm)
#np.save("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_GMV_spice_nrm.npy",san0_std_nrm)

san0_prfhrd_avg = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/SAN0/SAN0_array_GMVBHTTPRF_spice.npy").mean(axis=1)
san0_std_avg = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_GMV_spice.npy").mean(axis=1)
san0_prfhrd_nrm_avg = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4_prftsz/SAN0/SAN0_array_GMVBHTTPRF_spice_nrm.npy").mean(axis=1)
san0_std_nrm_avg = np.load("/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/gmv052425/gmvjtp_sep/crosstf_v5_lmin500_500_500_lmax3500_3000_3000_mmin100_binmaskcinv_notch_softinner_mtheta3_v4/SAN0/SAN0_array_GMV_spice_nrm.npy").mean(axis=1)

lmax = 4000
l = np.arange(0,lmax+1)
ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
slpp = np.insert(slpp, 0, 0)
slpp = np.insert(slpp, 0, 0)
clkk = slpp[:lmax+1] * (l*(l+1))**2/4

plt.figure(0)
plt.clf()
plt.plot(l, clkk, 'k', label='Fiducial $C_L^{\kappa\kappa}$')
plt.plot(l, san0_std_avg, color='firebrick', linestyle='-', alpha=0.8, label='SAN0, GMV Standard, Averaged')
plt.plot(l, n0_std[:lmax+1], color='darkblue', linestyle='-', alpha=0.8, label='N0, GMV Standard')
plt.plot(l, san0_std_nrm_avg, color='forestgreen', linestyle='-', alpha=0.8, label='SAN0, GMV Standard, Averaged, Normalized')
plt.plot(l, san0_prfhrd_avg, color='salmon', linestyle='--', alpha=0.8, label='SAN0, GMV Profile Hardening, Averaged')
plt.plot(l, n0_prfhrd[:lmax+1], color='cornflowerblue', linestyle='--', alpha=0.8, label='N0, GMV Profile Hardening')
plt.plot(l, san0_prfhrd_nrm_avg, color='lightgreen', linestyle='--', alpha=0.8, label='SAN0, GMV Profile Hardening, Averaged, Normalized')
plt.grid(True, linestyle="--", alpha=0.5)
#plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
plt.xlabel('$L$')
#plt.title(f"Sim {idx} SAN0",pad=10)
plt.title(f"Averaged SAN0",pad=10)
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-8,1e-7)
plt.tight_layout()
plt.savefig(f'/home/users/yukanaka/lensing_template/figs/san0_prfhrd.png',bbox_inches='tight')


