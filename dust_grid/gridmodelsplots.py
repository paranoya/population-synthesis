#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:56:52 2019

@author: pablo
"""

from matplotlib import pyplot as plt
import numpy as np

from galaxy_distrib_model import Model_grid

models = Model_grid(photomod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/',
specmod_path='population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/products/')


#%% 

plt.figure(figsize=(12, 9))

plt.subplot(351)
plt.title('u')
plt.imshow(models.u[7, :, :, 100], aspect='auto', origin='lower', 
           cmap='rainbow', extent=(models.extinction[0],models.extinction[-1],
                                   -0.7,1.7))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.ylabel(r'log($\frac{\tau}{Gyr})$')
plt.xlabel(r'E(B-V)')

plt.subplot(352)
plt.title('g')
plt.imshow(models.g[7, :, :, 100], aspect='auto',origin='lower', 
           cmap='rainbow', extent=(models.extinction[0],models.extinction[-1],
                                   -0.7,1.7))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()


plt.subplot(353)
plt.title('r')
plt.imshow(models.r[7, :, :, 100], aspect='auto',origin='lower', 
           cmap='rainbow', extent=(models.extinction[0],models.extinction[-1],
                                   -0.7,1.7))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(354)
plt.title('i')
plt.imshow(models.i[7, :, :, 100], aspect='auto', origin='lower', 
           cmap='rainbow', extent=(models.extinction[0],models.extinction[-1],
                                   -0.7,1.7))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(355)
plt.title('z')
plt.imshow(models.z[7, :, :, 100], aspect='auto', origin='lower', 
           cmap='rainbow', extent=(models.extinction[0],models.extinction[-1],
                                   -0.7,1.7))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(356)
plt.imshow(models.u[:, :, 0, 100], aspect='auto', origin='lower', 
           cmap='inferno_r', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'$\tau$ [Gyr]')
plt.ylabel(r'Z')

plt.subplot(357)
plt.imshow(models.g[:, :, 0, 100], aspect='auto', origin='lower', 
           cmap='inferno_r', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(358)
plt.imshow(models.r[:, :, 0, 100], aspect='auto', origin='lower', 
           cmap='inferno_r', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(359)
plt.imshow(models.i[:, :, 0, 100], aspect='auto', origin='lower', 
           cmap='inferno_r', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(3,5,10)
plt.imshow(models.z[:, :, 0, 100], aspect='auto', origin='lower', 
           cmap='inferno_r',
           extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(3,5,11)
plt.imshow(models.u[:, 150, :, 100], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'E(B-V)')
plt.ylabel(r'Z')

plt.subplot(3,5,12)
plt.imshow(models.g[:, 150, :, 100], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(3,5,13)
plt.imshow(models.r[:, 150, :, 100], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(3,5,14)
plt.imshow(models.i[:, 150, :, 100], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplot(3,5,15)
plt.imshow(models.z[:, 150, :, 100], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()

plt.subplots_adjust(wspace=1, hspace=0.3)


plt.savefig('magnitudes_param_dependence.png')

#%%

plt.figure(figsize=(10,6))
plt.subplot(2,4,1)
plt.title('D4000')
plt.imshow(models.lick_indices[7, -1, :, :], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.extinction[0],models.extinction[-1],
                   models.tau[0], models.tau[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.ylabel(r'$\tau$ [Gyr]')
plt.xlabel(r'E(B-V)')
plt.subplot(2,4,5)
plt.imshow(models.lick_indices[:, -1, :, 0], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'$\tau$ [Gyr]')
plt.ylabel(r'Z')


plt.subplot(2,4,2)
plt.title('Fe5270')
plt.imshow(models.lick_indices[:, -2, :, 0], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'$\tau$ [Gyr]')
plt.ylabel(r'Z')

plt.subplot(2,4,3)
plt.title('Mgb')
plt.imshow(models.lick_indices[:, -3, :, 0], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'$\tau$ [Gyr]')
plt.ylabel(r'Z')


plt.subplot(2,4,4)
plt.title(r'H$\beta$')
plt.imshow(models.lick_indices[:, -5, :, 0], aspect='auto', origin='lower', 
           cmap='gist_earth', extent=(models.tau[0],models.tau[-1],
                   models.metallicities[0], models.metallicities[-1]))
plt.grid(b=True, color='k', alpha=.4)
plt.colorbar()
plt.xlabel(r'$\tau$ [Gyr]')
plt.ylabel(r'Z')

plt.subplots_adjust(wspace=1, hspace=0.3)

plt.savefig('lick_gridmodels.png')


