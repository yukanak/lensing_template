import os,sys
from tqdm import tqdm
import argparse
import healpy as hp
import numpy as np
from healpy .rotator import Rotator

parser = argparse.ArgumentParser()
parser.add_argument('freq'    , default=None , type=int  , help='freq')
parser.add_argument('seed'    , default=None , type=int  , help='seed')
parser.add_argument('patch'   , default=None , type=int  , help='patch')
parser.add_argument('lmax'    , default=None , type=int  , help='lmax')
parser.add_argument('--nolog' , default=False, dest='nolog', action='store_true')

args  = parser.parse_args()
freq  = args.freq
seed  = args.seed
patch = args.patch
lmax  = args.lmax

# Rotation matrix
rotm = np.array([[0,0,0],
                 [0,180,0],
                 [180,180,0],
                 [0,180,180],
                 [0,60,90],
                 [0,120,90],
                 [0,180,90],
                 [0,240,90],
                 [0,300,90],
                 [0,360,90],
               ])
# Beam
beam = np.loadtxt('/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch/compiled_2020_beams_extended.txt')
if freq==95 : bl = beam[:lmax+1,1]
if freq==150: bl = beam[:lmax+1,2]
if freq==220: bl = beam[:lmax+1,3]

# Input fullsky map
# Primary CMB has to be for planck 2018 cosmology not MDPL2 cosmology. 
cmb = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed%d_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'%(seed),field=[0,1,2])

tot_t = cmb[0]
tot_q = cmb[1]
tot_u = cmb[2]

alm  = hp.map2alm([tot_t,tot_q,tot_u],lmax=lmax,use_pixel_weights=True)
# NOT APPLYING BEAM NOW
#alm[0] = hp.almxfl(alm[0],bl)
#alm[1] = hp.almxfl(alm[1],bl)
#alm[2] = hp.almxfl(alm[2],bl)

if freq==95:
    freq=90

dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/agora_input_skies_spt3g_patch/cmbonly/'
cmbname = 'cmbonly'

if patch==0:
    hp.write_alm(dir_out+'agora_%dghz_%s_rotated_%d%d.alm'%(freq,cmbname,seed,patch),alm)

else:
    r    = Rotator(rot=[rotm[patch][0],rotm[patch][1],rotm[patch][2]], deg=True,inv=True)
    ralm = r.rotate_alm(alm)  
    hp.write_alm(dir_out+'agora_%dghz_%s_rotated_%d%d.alm'%(freq,cmbname,seed,patch),ralm)


