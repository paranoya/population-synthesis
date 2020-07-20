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
import photometry 
import lick_indices


from dust_extinction import DustModelGrid


# =============================================================================
# Select model SED folder
# =============================================================================

#SEDfolder = '../../population_synthesis/delayedtau_burst/epoch13.648Gyr/SED_kro*'

## Bruzual & Charlote
#SEDfolder = '../../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr/SED_cha*'

## Bruzual & Charlote EXP-ALPHA
SEDfolder = '../../population_synthesis/EXP_delayed_tau_BC/epoch13.7Gyr_long/SED_cha*'

# Bruzual & Charlote QUENCHING
#SEDfolder = '../../population_synthesis/tau_delayedEXPSFR_BC_quenched/quenching_epoch_13.348Gyr/SED_cha*'

# PopStar
#SEDfolder = '../../population_synthesis/tau_delayedEXPSFR/epoch13.648Gyr/SED_cha*'

SEDpaths = np.sort(glob(SEDfolder))
print('Number of input SEDs: {}'.format(len(SEDpaths)))
# =============================================================================
# Output
# =============================================================================

#output_path = '../../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr' \
#                +'/products'
#output_path = '../../population_synthesis/tau_delayedEXPSFR_BC_quenched/quenching_epoch_13.348Gyr' \
#                +'/products'
output_path = '../../population_synthesis/EXP_delayed_tau_BC/epoch13.7Gyr_long' \
                +'/products'
                
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
activate_photometry = True
#photometry = False


photo_system= 'AB'            
#photo_system= 'Vega'            

# =============================================================================
# Select Lick indices (Set to false if not needed)
# =============================================================================

activate_lick_idx = True
#lick_idx = [
#             'Lick_Hd_A', 
#             'Lick_Hg_A', 
#             'Lick_Fe4383',
#             'Lick_Fe5270',
#             'Lick_Fe5335',                          
#             'Lick_Mg1',
#             'Lick_Mgb'
#                   
#            ]
lick_idx = np.loadtxt('../lick_list.txt', usecols=0, dtype=str)


#lick_idx = False

balmer_break = True
D4000 = 'D4000'       # For output files
#balmer_break = False
#D4000 = ''


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
#%% Computation without dust models
# =============================================================================
# =============================================================================
# =============================================================================
                
if dust_grid ==False:                
    print('---> Starting computation without dust grid')
    for SEDpath in SEDpaths:
        print(SEDpath)
        wavelength, flux = np.loadtxt(SEDpath, usecols=(0, 4), unpack=True) #TODO: select which column to read
        
        if (activate_photometry)&(activate_lick_idx):
            
            photometric_data= []
            lick_data = []
            
            if len(lick_idx)>len(photo_bands):
                primary_loop_length = len(lick_idx)
               
                for element in range(primary_loop_length):
                
                    l_idx  = lick_indices.Lick_index(flux, wavelength, 
                                                     lick_idx[element])
                    
                    lick_data.append(l_idx.lick_index)
                    
                    if element<len(photo_bands):
                        mag = photometry.magnitude(absolute=True, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                        photometric_data.append(mag.magnitude)
                if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    
            else:
                primary_loop_length = len(lick_idx)
                
                
                for element in range(primary_loop_length):
                
                    mag = photometry.magnitude(absolute=True, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                    photometric_data.append(mag.magnitude)
                    
                    if element<len(lick_idx):
                        
                        l_idx  = lick_indices.Lick_index(flux, wavelength, 
                                                     lick_idx[element])
                        
                        lick_data.append(l_idx.lick_index)
        
                if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    
        elif photo_bands==False:
            
            lick_data = []
            
            for element in range(len(lick_idx)):
                
                    l_idx  = lick_indices.Lick_index(flux, wavelength, 
                                                     lick_idx[element])
                    
                    lick_data.append(l_idx.lick_index)
            
            if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    
        else:
            
            photometric_data= []
            
            for element in range(len(photo_bands)):
                
                    mag = photometry.magnitude(absolute=True, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                    photometric_data.append(mag.magnitude)
    
    
        # Writing output files
        # TODO: Modify the way of assigning the file names. Currently hardcoded.
        # TODO: Adaptative file content (for cases with no photometry/lick indices)
        with open(output_path+'lick_indices_'+'Z_'+SEDpath[-22:-16]\
                  +'.txt', '+a') as txt:
            txt.write(SEDpath[-11:-4]+'   ')
            for i in range(len(lick_data)):
                txt.write('   {:}   '.format(lick_data[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        with open(output_path+'photometry_'+'Z_'+SEDpath[-22:-16]\
                 +'.txt', '+a') as txt:
            txt.write(SEDpath[-11:-4]+'   ')
            for i in range(len(photometric_data)):
                txt.write('{:}   '.format(photometric_data[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        f = open(output_path+'lick_indices_'+'Z_'+SEDpath[-22:-16]\
                  +'.txt','r+')
        lines = f.readlines() # read old content
        f.seek(0) # go back to the beginning of the file
        f.write('# tau [Gyrs] '+str(lick_idx)+ D4000) 
        f.close()        
    
        f = open(output_path+'photometry_'+'Z_'+SEDpath[-22:-16]\
                      +'.txt','r+')
        lines = f.readlines() # read old content
        f.seek(0) # go back to the beginning of the file
        f.write('# tau [Gyrs] ' +str(photo_bands)) 
        f.close()                
        
# =============================================================================
# =============================================================================
# =============================================================================
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
        
        fluxgrid = dustgrid.SED_grid
        A_v = dustgrid.A_V
        
        for j_element in range(fluxgrid.shape[1]) :
            
            flux = fluxgrid[:, j_element]
            A_v_i = A_v[j_element]
            
            if (activate_photometry)&(activate_lick_idx):
                
                photometric_data= []
                lick_data = []
                
                if len(lick_idx)>len(photo_bands):
                    primary_loop_length = len(lick_idx)
                   
                    for element in range(primary_loop_length):
                    
                        l_idx  = lick_indices.Lick_index(
                                                    flux=flux, 
                                                    lamb=wavelength, 
                                                    flux_err=np.zeros_like(flux),
                                                    lick_index_name=lick_idx[element])
                        
                        lick_data.append(l_idx.lick_index)
#                        l_idx.plot_index(data=lick_idx[element], folder=lick_idx[element]+'.png')
                        if element<len(photo_bands):
                            mag = photometry.magnitude(absolute=True, 
                                                   filter_name=photo_bands[element], 
                                                   wavelength=wavelength, flux=flux,
                                                   photometric_system=photo_system)
                            photometric_data.append(mag.magnitude)
                    
                    if balmer_break==True:
                        bbreak = lick_indices.BalmerBreak(flux, wavelength)
                        lick_data.append(bbreak.D4000)
                    
                else:
                    primary_loop_length = len(lick_idx)
                    
                    
                    for element in range(primary_loop_length):
                    
                        mag = photometry.magnitude(absolute=True, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                        
                        photometric_data.append(mag.magnitude)
                        
                        if element<len(lick_idx):
                            
                            l_idx  = lick_indices.Lick_index(flux, wavelength, 
                                                         lick_idx[element])
                            
                            lick_data.append(l_idx.lick_index)
            
                    if balmer_break==True:
                        bbreak = lick_indices.BalmerBreak(flux, wavelength)
                        lick_data.append(bbreak.D4000)
            
            elif photo_bands==False:
                
                lick_data = []
                
                for element in range(len(lick_idx)):
                    
                        l_idx  = lick_indices.Lick_index(
                                flux=flux, 
                                lamb=wavelength, 
                                flux_err=np.zeros_like(flux),
                                lick_index_name=lick_idx[element])
                        
                        lick_data.append(l_idx.lick_index)
                
                if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
        
            else:
                
                photometric_data= []
                
                for element in range(len(photo_bands)):
                    
                        mag = photometry.magnitude(absolute=True, 
                                                   filter_name=photo_bands[element], 
                                                   wavelength=wavelength, flux=flux,
                                                   photometric_system=photo_system)
                        photometric_data.append(mag.magnitude)

# =============================================================================
# FOR MODELS WITH SINGLE METALLICITY
# =============================================================================
  
            # Writing output files
            # TODO: Modify the way of assigning the file names. Currently hardcoded.
#            if activate_lick_idx:
#                with open(output_path+'lick_indices_'+'Z_'+SEDpath[-22:-16]\
#                          +'.txt', '+a') as txt:                
#                    txt.write(SEDpath[-11:-4]+'   ')
#                    txt.write('   {:}   '.format(A_v_i))
#                    for i in range(len(lick_data)):
#                        txt.write('   {:}   '.format(lick_data[i]))
#                    txt.write('\n') 
#                txt.close
#                del txt
#            if activate_photometry:
#                with open(output_path+'photometry_'+'Z_'+SEDpath[-22:-16]\
#                     +'.txt', '+a') as txt:                
#                    txt.write(SEDpath[-11:-4]+'   ')
#                    txt.write('   {:}   '.format(A_v_i))
#                    for i in range(len(photometric_data)):
#                        txt.write('{:}   '.format(photometric_data[i]))
#                    txt.write('\n') 
#                txt.close
#            del txt
            
# =============================================================================
# FOR MODELS WITH CHEMICAL EVOLUTION
# =============================================================================
            alpha=SEDpath[-12:-4]
            if activate_lick_idx:
                with open(output_path+'lick_indices.txt', '+a') as txt:                
                    txt.write(alpha+'   ')
                    txt.write('   {:}   '.format(A_v_i))
                    for i in range(len(lick_data)):
                        txt.write('   {:}   '.format(lick_data[i]))
                    txt.write('\n') 
                txt.close
                del txt       
            if activate_photometry:
                with open(output_path+'photometry.txt', '+a') as txt:                
                    txt.write(alpha+'   ')
                    txt.write('   {:}   '.format(A_v_i))
                    for i in range(len(photometric_data)):
                        txt.write('{:}   '.format(photometric_data[i]))
                    txt.write('\n') 
                txt.close
                del txt       
# =============================================================================
# Info file    
# =============================================================================
if dust_grid:    
    with open(output_path+'photometry_info.txt', 'w') as txt:
            txt.write('# photometry_(met).txt file content\n')
            txt.write('tau  Av  ')                  
            for band in photo_bands:    
                          txt.write(band+'  ')                      
    with open(output_path+'lick_indices_info.txt', 'w') as txt:
            txt.write('# lick_indices_(met).txt file content\n')
            txt.write('tau  Av  ')                  
            for lick in lick_idx:    
                          txt.write(lick+'  ')                      
            if balmer_break:
                txt.write('D4000')
                                               
else:    
    with open(output_path+'photometry_info.txt', 'w') as txt:
            txt.write('# photometry_(met).txt file content\n')
            txt.write('tau  ')                  
            for band in photo_bands:    
                          txt.write(band+'  ')                      
    
    with open(output_path+'lick_indices_info.txt', 'w') as txt:
            txt.write('# lick_indices_(met).txt file content\n')
            txt.write('tau  ')                  
            for lick in lick_idx:    
                          txt.write(lick+'  ')                      
            if balmer_break:
                txt.write('D4000')              
                
                
# Mr. Krtxo...                
