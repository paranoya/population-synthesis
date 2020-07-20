#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:04:39 2019

@author: pablo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:30:45 2019

@author: pablo
"""

from glob import glob
import os

import numpy as np

import sys
sys.path.append("..")
sys.path.append("../..")
import photometry 
import lick_indices

from dust_extinction import DustModelGrid
import astropy.io.fits as fits

# =============================================================================
# Select model SED folder
# =============================================================================

SEDfolder = '../../data/MANGA/*.fits'


SEDpaths = np.sort(glob(SEDfolder))

SEDpaths = [SEDpaths[-1]]

cube = fits.open(SEDpaths[0])
flux = cube[1].data
flux_err = 1/cube[2].data
wavelength = cube[6].data
#wl = wl /(1+z_redshift)
cube.close()

# =============================================================================
# Select photometric bands and photometric system (Set to false if not needed)
# =============================================================================

photo_bands = [
               'u',
               'g',
               'r',
#               'i',
#               'z'
               ]
#photo_bands = False

photo_system= 'AB'            
#photo_system= 'Vega'            

# =============================================================================
# Select Lick indices (Set to false if not needed)
# =============================================================================

lick_idx = [
#             'HdeltaA', 
#             'HdeltaF',
#             'Ca4227',
#             'HgammaA', 
#             'HgammaF',
             'Fe4383',
#             'Ca4455',
             'Hbeta', 
             'Mg1',
             'Mgb',
             'Fe5270'
            ]
#lick_idx = False

balmer_break = True
D4000 = 'D4000'       # For output files

#balmer_break = False
#D4000 = ''

# =============================================================================
# Dust grid models
# =============================================================================

#dust_grid = True
dust_grid = False

ext_law ='calzetti2000'
#ext_law ='cardelli89'

n_models = 10

# =============================================================================
# Output
# =============================================================================

output_path = '../../products/MANGA'
if os.path.isdir(output_path)==False:
    os.mkdir(output_path)
    print('Directoy created : \n', output_path)
                
                

# =============================================================================
# =============================================================================
# =============================================================================
#%% Computation without dust models
# =============================================================================
# =============================================================================
# =============================================================================
photometry_map = np.empty((len(photo_bands), flux.shape[1], flux.shape[2]))                
lick_map = np.empty((len(lick_idx)+1, flux.shape[1], flux.shape[2]))                
lick_err_map = np.empty((len(lick_idx)+1, flux.shape[1], flux.shape[2]))                

if dust_grid ==False:                
    for i_elem in range(flux.shape[1]):
        for j_elem in range(flux.shape[2]):
#        i_elem=25
#        j_elem=25
            print('Computing region ({},{})'.format(i_elem, j_elem))
            flux_ij = flux[:, i_elem, j_elem]
            flux_ij_err = flux_err[:, i_elem, j_elem]
                
            flux_ij = flux_ij*wavelength*1e-17 # erg/s/cm2
            flux_ij_err = flux_ij_err*wavelength*1e-17 # erg/s/cm2
            
            if (len(photo_bands)!=0)&(len(lick_idx)!=0):
                            
                if len(lick_idx)>len(photo_bands):
                    primary_loop_length = len(lick_idx)
                   
                    for element in range(primary_loop_length):
                    
                        l_idx  = lick_indices.Lick_index(
                                flux=flux_ij, 
                                flux_err=flux_ij_err,
                                lamb=wavelength, 
                                lick_index_name= lick_idx[element])
    
                        lick_map[element, i_elem, j_elem] = l_idx.lick_index
                        lick_err_map[element, i_elem, j_elem] = l_idx.lick_index_err
                        
                        if element<len(photo_bands):
                            mag = photometry.magnitude(absolute=False, 
                                                   filter_name=photo_bands[element], 
                                                   wavelength=wavelength, flux=flux_ij,
                                                   photometric_system=photo_system)
                            photometry_map[element, i_elem, j_elem] = mag.magnitude
                            
                    if balmer_break==True:
                        bbreak = lick_indices.BalmerBreak(flux_ij, wavelength)
                        lick_map[-1, i_elem, j_elem] = bbreak.D4000
                        lick_err_map[-1, i_elem, j_elem] = bbreak.sigma_D4000
                        
                else:
                    primary_loop_length = len(photo_bands)
                    
                    
                    for element in range(primary_loop_length):
                    
                        mag = photometry.magnitude(absolute=False, 
                                                   filter_name=photo_bands[element], 
                                                   wavelength=wavelength, flux=flux_ij,
                                                   photometric_system=photo_system)
                        photometry_map[element, i_elem, j_elem] = mag.magnitude
                        
                        if element<len(lick_idx):
                            
                            l_idx  = lick_indices.Lick_index(
                                flux=flux_ij, 
                                flux_err=flux_ij_err,
                                lamb=wavelength, 
                                lick_index_name= lick_idx[element])
                            
                            lick_map[element, i_elem, j_elem] = l_idx.lick_index
                            lick_err_map[element, i_elem, j_elem] = l_idx.lick_index_err
                        
                    if balmer_break==True:
                        bbreak = lick_indices.BalmerBreak(flux_ij, wavelength)
                        lick_map[-1, i_elem, j_elem] = bbreak.D4000
                        lick_err_map[-1, i_elem, j_elem] = bbreak.sigma_D4000
            elif photo_bands==False:
                
                
                for element in range(len(lick_idx)):
                    
                        l_idx  = lick_indices.Lick_index(
                                flux=flux_ij, 
                                flux_err=flux_ij_err,
                                lamb=wavelength, 
                                lick_index_name= lick_idx[element])
    
                        lick_map[element, i_elem, j_elem] = l_idx.lick_index
                        lick_err_map[element, i_elem, j_elem] = l_idx.lick_index_err
                        
                if balmer_break==True:
                        bbreak = lick_indices.BalmerBreak(flux_ij, wavelength)
                        lick_map[-1, i_elem, j_elem] = bbreak.D4000
                        lick_err_map[-1, i_elem, j_elem] = bbreak.sigma_D4000
            else:
                
                photometric_data= []
                
                for element in range(len(photo_bands)):
                    
                        mag = photometry.magnitude(absolute=False, 
                                                   filter_name=photo_bands[element], 
                                                   wavelength=wavelength, flux=flux_ij,
                                                   photometric_system=photo_system)
                        photometry_map[element, i_elem, j_elem] = mag.magnitude
        
        
        
# =============================================================================
# =============================================================================
# =============================================================================
