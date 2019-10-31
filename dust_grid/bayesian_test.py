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

from galaxy_distrib_model import Model_grid


GAMAphotometry = pd.read_csv('GAMA_data/GAMAsample_totalphotometry.csv')
u = GAMAphotometry['Petro_u_abs']
u_err = GAMAphotometry['Petro_u_err']
g = GAMAphotometry['Petro_g_abs']
g_err = GAMAphotometry['Petro_g_err']
r = GAMAphotometry['Petro_r_abs']
r_err = GAMAphotometry['Petro_r_err']
i = GAMAphotometry['Petro_i_abs']
i_err = GAMAphotometry['Petro_i_err']
z = GAMAphotometry['Petro_z_abs']
z_err = GAMAphotometry['Petro_z_err']


GAMAlick = pd.read_csv('GAMA_data/products/GAMA_lick_indices.csv')

lick_indices = []
lick_indices_err = []
lick_idx = [
             'HdeltaA', 
             'HdeltaF',
             'Ca4227',
             'HgammaA', 
             'HgammaF',
             'Fe4383',
             'Ca4455',
             'Hbeta', 
             'Mg1',
             'Mgb',
             'Fe5270',
             'D4000'
            ]

for i_elem in range(len(lick_idx)):
    lick_indices.append(GAMAlick[lick_idx[i_elem]])
    lick_indices_err.append(GAMAlick[lick_idx[i_elem]+'_err'])

lick_indices = np.array(lick_indices)
lick_indices_err = np.array(lick_indices_err)

del GAMAphotometry

red_gals = np.where(u-r>2.5)[0]
blue_gals = np.where(u-r<1.5)[0]


#%%
models = Model_grid(photomod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/',
specmod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/')

met, taus, extinction, mass, likelihood = models.bayesian_model_assignment(u=u, g=g, r=r, i=i, z=z,
                                                   u_err = u_err, g_err=g_err,
                                                   r_err=r_err, i_err=i_err,
                                                   z_err=z_err, 
                                                   lick_indices=lick_indices,
                                                   lick_indices_err = lick_indices_err
                                                   )


df = pd.DataFrame({'Tau':taus,
                  'Met_Z':met,
                  'E(B-V)':extinction,
                  'Mass':mass
                  })

df.to_csv('bayesian_assignment_GAMA.csv')

#%%

plt.figure(figsize=(10,10))
plt.scatter(taus, met/0.02,  s=0.3)
plt.colorbar()
plt.savefig('tau_met.png')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(taus, u-r, c=extinction,  s=0.3)
plt.colorbar()
plt.savefig('tau_u_r.png')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(u-r, 3.1*extinction, )
plt.savefig('u_r_ext.png')
plt.close()

#%%
#fig = plt.figure()
#
#dpdlogtau= np.sum(np.exp(-
#                         ((u[1]-models.u)**2/(u_err[1]**2)+
#                          (g[1]-models.g)**2/(g_err[1]**2)+
#                          (r[1]-models.r)**2/(r_err[1]**2)+
#                          (i[1]-models.i)**2/(i_err[1]**2)+                          
#                          (z[1]-models.z)**2/(z_err[1]**2))
#                         ), axis=(0, 2, 3))
#
#dpdlogz= np.sum(np.exp(-
#                         ((u[1]-models.u)**2/(u_err[1]**2)+(u[1]-models.u)**2/(u_err[1]**2)+
#                          (g[1]-models.g)**2/(g_err[1]**2)+(g[1]-models.g)**2/(g_err[1]**2)+
#                          (r[1]-models.r)**2/(r_err[1]**2)+(r[1]-models.r)**2/(r_err[1]**2)+
#                          (i[1]-models.i)**2/(i_err[1]**2)+(i[1]-models.i)**2/(i_err[1]**2)+                          
#                          (z[1]-models.z)**2/(z_err[1]**2)+(z[1]-models.z)**2/(z_err[1]**2))
#                         ), axis=(1, 2, 3))
#
##plt.plot(models.tau, dpdlogtau/np.sum(dpdlogtau))
#
#plt.loglog(models.tau, dpdlogtau/np.sum(dpdlogtau))
#plt.plot(models.metallicities, dpdlogz/np.sum(dpdlogz))
#print(taus,'\n', met,'\n', extinction,'\n', mass)

xlick = -1
ylick = -3

plt.figure()
for i_elem in range(0,25,2):
    plt.plot(models.lick_indices[i_elem,xlick,:,0], 
             models.lick_indices[i_elem,ylick,:,0])
for i_elem in range(100):
    plt.errorbar(lick_indices[xlick,i_elem], lick_indices[ylick,i_elem],
                 xerr=lick_indices_err[xlick,i_elem], 
                 yerr=lick_indices_err[ylick,i_elem],
                 color='k')

plt.xlim(0,4)
plt.ylim(0,6)




















