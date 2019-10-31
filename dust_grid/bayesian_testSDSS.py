#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:59:16 2019

@author: pablo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:24:42 2019

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors 
import pandas as pd
from astropy.io import fits
from galaxy_distrib_model import Model_grid

import sys 
sys.path.append('products/')


from kcorrection import calc_kcor as Kcor
from astropy.cosmology import FlatLambdaCDM
from astropy import units

cosmo = FlatLambdaCDM(H0=100 * units.km / units.s / units.Mpc, Tcmb0=2.725 * units.K, Om0=0.3)

#%%
SDSShighSN = fits.open('SDSS_data/MyTable_lick_idx051019_alllick_highSN_PabloCorcho.fit')

data = SDSShighSN[1].data
u = data['petroMag_u']
g = data['petroMag_g']
r = data['petroMag_r']
i = data['petroMag_i']
z = data['petroMag_z']

red_z = data['Z']

u_err = data['petroMagErr_u']
g_err = data['petroMagErr_g']
r_err = data['petroMagErr_r']
i_err = data['petroMagErr_i']
z_err = data['petroMagErr_z']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# K-correction    
K_u_corr = []
K_g_corr = []
K_r_corr = []
K_i_corr = []
K_z_corr = []

for j in range(len(u)):
   K_u_corr.append( Kcor('u',   red_z[j], 'u - r', u[j]-r[j]) ) 
   K_g_corr.append( Kcor('g', red_z[j], 'g - r', g[j]-r[j]) ) 
   K_r_corr.append( Kcor('r', red_z[j], 'g - r', g[j]-r[j]) ) 
   K_i_corr.append( Kcor('i', red_z[j], 'g - i', g[j]-i[j]) ) 
   K_z_corr.append( Kcor('z', red_z[j], 'r - z', r[j]-z[j]) ) 
   
#Hubble_cte = 100 # Km/s/Mpc
#phot_distance = (red_z*3e5/Hubble_cte) *1e6   # pc  


phot_distance = cosmo.luminosity_distance(red_z).value*1e6 # expressed in pc

u_abs = u - 5*(np.log10(phot_distance)-1) -K_u_corr
g_abs = g - 5*(np.log10(phot_distance)-1)   -K_g_corr # M_r - 5log(h)
r_abs = r - 5*(np.log10(phot_distance)-1)  - K_r_corr 
i_abs = i - 5*(np.log10(phot_distance)-1)  - K_i_corr 
z_abs = z - 5*(np.log10(phot_distance)-1)  - K_z_corr 

u_abs2 = u - 5*(np.log10(phot_distance)-1) 
g_abs2 = g - 5*(np.log10(phot_distance)-1)  
r_abs2 = r - 5*(np.log10(phot_distance)-1)  
i_abs2 = i - 5*(np.log10(phot_distance)-1)  
z_abs2 = z - 5*(np.log10(phot_distance)-1)  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lick_indices = [
                data['lick_hd_a_sub'], 
                data['lick_hd_f_sub'], 
                data['lick_ca4227_sub'], 
                data['lick_hg_a_sub'], 
                data['lick_hg_f_sub'], 
                data['lick_fe4383_sub'], 
                data['lick_ca4455_sub'], 
                data['lick_hb_sub'], 
                data['lick_mg1_sub'], 
                data['lick_mgb_sub'], 
                data['lick_fe5270_sub'], 
                data['d4000_sub'] 
                ]
lick_indices = np.array(lick_indices)

lick_indices_err = [
                data['lick_hd_a_sub_err'], 
                data['lick_hd_f_sub_err'], 
                data['lick_ca4227_sub_err'], 
                data['lick_hg_a_sub_err'], 
                data['lick_hg_f_sub_err'], 
                data['lick_fe4383_sub_err'], 
                data['lick_ca4455_sub_err'], 
                data['lick_hb_sub_err'], 
                data['lick_mg1_sub_err'], 
                data['lick_mgb_sub_err'], 
                data['lick_fe5270_sub_err'], 
                data['d4000_sub_err'] 
                ]

lick_indices_err = np.array(lick_indices_err)


tau_v = data['tauv_cont']   # V-band optical depth (TauV = A_V / 1.086) affecting the stars from best fit model (best of 4 Z's)


#
#red_gals = np.where(u-r>2.5)[0]
#blue_gals = np.where(u-r<1.5)[0]

#u=np.array(u[red_gals]) 
#g=np.array(g[red_gals])
#r=np.array(r[red_gals])
#i=np.array(i[red_gals])
#z=np.array(z[red_gals])
#lick_idx = [
#             'HdeltaA', 
#             'HdeltaF',
#             'Ca4227',
#             'HgammaA', 
#             'HgammaF',
#             'Fe4383',
#             'Ca4455',
#             'Hbeta', 
#             'Mg1',
#             'Mgb',
#             'Fe5270'
#             'd4000'
#            ]

#%% 
models = Model_grid(photomod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/',
specmod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/')

met, taus, extinction, mass, likelihood_lick, likelihood = models.bayesian_model_assignment(u=u_abs, 
                                                               g=g_abs,
                                                               r=r_abs,
                                                               i=i_abs,
                                                               z=z_abs,
                                                   u_err = u_err,
                                                   g_err=g_err,
                                                   r_err=r_err,
                                                   i_err=i_err,
                                                   z_err=z_err, 
                                           lick_indices=lick_indices,
                                 lick_indices_err = lick_indices_err
                                                   )


df = pd.DataFrame({'Tau':taus,
                  'Met_Z':met,
                  'E(B-V)':extinction,
                  'Mass':mass
                  })

df.to_csv('bayesian_assignment_SDSS_galaxies.csv')



#%%
# =============================================================================
# 
# =============================================================================


a = likelihood_lick[2]
b = likelihood[2]

#lhood = np.exp(-np.sum(a, axis=2))
lhood = np.exp(-a)
lhood_t = np.exp(-b)


plt.figure()
plt.imshow(np.sum(lhood, axis=2), aspect='auto',
           extent=(models.tau[0], models.tau[-1], 
                   models.metallicities[0], models.metallicities[-1]))
plt.colorbar()

plt.figure()
plt.imshow(lhood_t[:,:,0,0], aspect='auto',
           extent=(models.tau[0], models.tau[-1], 
                   models.metallicities[0], models.metallicities[-1]))
plt.colorbar()

join_lh = np.exp(-(a[:,:,:, np.newaxis]+b))

plt.figure()
plt.imshow(join_lh[:,:,0,0], aspect='auto',
           extent=(models.tau[0], models.tau[-1], 
                   models.metallicities[0], models.metallicities[-1]))
plt.colorbar()


#%%

xlick = -1
ylick = 3

plt.figure(figsize=(10,10))
for i_elem in range(25):
    plt.plot(models.lick_indices[i_elem,xlick,:,0], 
             models.lick_indices[i_elem,ylick,:,0])
for i_elem in range(0,8000, 50):
    plt.errorbar(lick_indices[xlick,i_elem], lick_indices[ylick,i_elem],
                 xerr=lick_indices_err[xlick,i_elem], 
                 yerr=lick_indices_err[ylick,i_elem],
                 color='k')
plt.xlim(0.5,2.5)
plt.ylim(-8,6)


young = np.where(lick_indices[-1,:]<1.2)[0]


