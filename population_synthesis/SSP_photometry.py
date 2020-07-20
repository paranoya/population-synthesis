#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:30:28 2020

@author: pablo
"""
from matplotlib import pyplot as plt

from glob import glob

import numpy as np

import os
import sys
sys.path.append("..")
sys.path.append("../products_codes")
import photometry 

from dust_extinction_model_grid import DustModelGrid
import SSP 

import units

# %%

my_ssp = SSP.BC03_Padova94(mode='hr', IMF='chab')   
# my_ssp = SSP.PopStar(IMF='cha_0.15_100')   

ages = my_ssp.ages[0:-1:10]
met = my_ssp.metallicities

wavelength = my_ssp.wavelength*1e10
nu = units.c/wavelength/1e-10

# SEDS = my_ssp.SED[:, 0:-1:10, :]*units.Msun*wavelength #erg/s
SEDS = my_ssp.SED[:, 0:-1:10, :]/units.L_sun * units.Angstrom / units.erg * units.Msun #erg/s
SEDS *= units.L_sun * wavelength

SEDS.resize((SEDS.shape[0]*SEDS.shape[1], SEDS.shape[2]))

plt.figure()
plt.loglog(wavelength, SEDS[30, :])

# %%
# =============================================================================
# Output
# =============================================================================


output_path = 'data/BC03/photometry'
                
if os.path.isdir(output_path)==False:
    os.mkdir(output_path)
    print('Directoy created : \n', output_path)

output_path = output_path+'/'

# =============================================================================
# Select photometric bands and photometric system (Set to false if not needed)
# =============================================================================

photo_bands = [
               'GFUV',
               'GNUV',
               'u',
               'g',
               'r',
               'i',
               'z',
               '2MASS_J',
               '2MASS_H',
               '2MASS_Ks'
               ]

# =============================================================================
# Dust grid models
# =============================================================================

dust_grid = True
ext_law ='calzetti2000'
#ext_law ='cardelli89'

n_models = 30

        
# =============================================================================
# =============================================================================
# =============================================================================
#%% Computation with dust models
# =============================================================================
# =============================================================================
# =============================================================================

all_models = []

if dust_grid==True:        
    print('---> Starting computation with dust grid')
    for ith in range(SEDS.shape[0]):
        print(ith)
        flux = SEDS[ith, :]
        
        dustgrid = DustModelGrid(flux=flux, wavelength=wavelength, 
                                 ext_law_name=ext_law,
                                 dust_dimension = n_models)
        
        fluxgrid = dustgrid.SED_grid
        A_v = dustgrid.A_V
        
        all_grids = []
        for extinc_i in range(fluxgrid.shape[1]) :
            
            flux = fluxgrid[:, extinc_i]
            A_v_i = A_v[extinc_i]
                
            photometric_data= []
                
            for filt_i in range(len(photo_bands)):
                mag = photometry.magnitude(absolute=True, 
                                       filter_name=photo_bands[filt_i], 
                                       wavelength=wavelength, flux=flux)
                photometric_data.append(mag.band_flux())
            all_grids.append(photometric_data)
            
        all_models.append(all_grids)
# %%

all_models = np.array(all_models)

all_models = all_models.reshape(met.size, ages.size, n_models, len(photo_bands))


np.save('photometry_models.npy', all_models)    
np.save('photometry_models_extinction.npy', A_v)    

# Mr. Krtxo...                