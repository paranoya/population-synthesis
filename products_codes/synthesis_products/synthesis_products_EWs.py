#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:04:35 2019

@author: pablo
"""

from glob import glob

import numpy as np

import os
import sys
sys.path.append("..")
sys.path.append("../..")

from model_equivalent_width import model_equivalent_width as mod_EW

from dust_extinction import DustModelGrid


# =============================================================================
# Select model SED folder
# =============================================================================

#SEDfolder = '../../population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/SED_kro*'

# Bruzual & Charlote
SEDfolder = '../../population_synthesis/tau_delayedEXPSFR_BC_quenched/quenching_epoch_13.647Gyr/SED_cha*'

# PopStar
#SEDfolder = '../../population_synthesis/tau_delayedEXPSFR/epoch13.648Gyr/SED_kro*'

SEDpaths = np.sort(glob(SEDfolder))
print('Number of input SEDs {}'.format(len(SEDpaths)))
# =============================================================================
# Output
# =============================================================================

output_path = '../../population_synthesis/tau_delayedEXPSFR_BC_quenched/quenching_epoch_13.647Gyr/products'

if os.path.isdir(output_path)==False:
    os.mkdir(output_path)
    print('Directoy created : \n', output_path)

output_path = output_path+'/'





equivalent_widths = ['halpha', 'hbeta']


# =============================================================================
# Dust grid models
# =============================================================================

dust_grid = True
ext_law ='calzetti2000'
#ext_law ='cardelli89'

n_models = 30

                

#%% Computation with dust models
# =============================================================================
# =============================================================================
# =============================================================================

if dust_grid==True:        
    print('---> Starting computation with dust grid')
    for SEDpath in SEDpaths:
        print(SEDpath)
        wavelength, flux = np.loadtxt(SEDpath, usecols=(0, 4), unpack=True) #TODO: select which column to read
        
        dustgrid = DustModelGrid(flux=flux, wavelength=wavelength, 
                                 ext_law_name=ext_law,
                                 dust_dimension = n_models)
        Z = float(SEDpath[-23:-17])
        fluxgrid = dustgrid.SED_grid
        A_v = dustgrid.A_V
        
        for j_element in range(fluxgrid.shape[1]) :
            
            flux = fluxgrid[:, j_element]
            A_v_i = A_v[j_element]
            
            
            
            EW_Hbeta = mod_EW(flux=flux, wavelength=wavelength, Z=Z, line='hbeta')
            EW_Halpha = mod_EW(flux=flux, wavelength=wavelength, Z=Z, line='halpha')
            
#            print(EW_Halpha.EW)
#            print(EW_Halpha.Q)
            # Writing output files
            # TODO: Modify the way of assigning the file names. Currently hardcoded.
            
            with open(output_path+'equivalent_widths_'+'Z_'+SEDpath[-22:-16]\
                      +'.txt', '+a') as txt:
                txt.write(SEDpath[-11:-4]+'   ')
                txt.write('   {:}   {:}   {:}'.format(A_v_i, 
                          EW_Hbeta.EW, 
                          EW_Halpha.EW))                
                txt.write('\n') 
            txt.close
            del txt
            
