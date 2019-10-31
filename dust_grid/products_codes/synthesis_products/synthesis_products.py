#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:04:35 2019

@author: pablo
"""

from glob import glob

import numpy as np

import sys
sys.path.append("..")
sys.path.append("../..")
import photometry 
import lick_indices

from dust_extinction_model_grid import DustModelGrid


# =============================================================================
# Select model SED folder
# =============================================================================

SEDfolder = '../../population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/SED_kro*'


SEDpaths = np.sort(glob(SEDfolder))


# =============================================================================
# Select photometric bands and photometric system (Set to false if not needed)
# =============================================================================

photo_bands = [
               'u',
               'g',
               'r',
               'i',
               'z'
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
#             'Fe4383',
#             'Ca4455',
             'Hbeta', 
             'Mg1',
             'Mgb',
             'Fe5270',
             'Fe5335',
             'DTT_CaII8498'
            ]
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
# Output
# =============================================================================

output_path = '../../population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr' \
                +'/products/'
                
                

# =============================================================================
# =============================================================================
# =============================================================================
#%% Computation without dust models
# =============================================================================
# =============================================================================
# =============================================================================
                
if dust_grid ==False:                
    for SEDpath in SEDpaths:
        print(SEDpath)
        wavelength, flux = np.loadtxt(SEDpath, usecols=(0, 4), unpack=True) #TODO: select which column to read
        
        
        if (len(photo_bands)!=0)&(len(lick_idx)!=0):
            
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
        with open(output_path+'lick_indices_'+'Z_'+SEDpath[-24:-18]\
                  +'.txt', '+a') as txt:
            txt.write(SEDpath[-13:-4]+'   ')
            for i in range(len(lick_data)):
                txt.write('   {:}   '.format(lick_data[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        with open(output_path+'photometry_'+'Z_'+SEDpath[-24:-18]\
                 +'.txt', '+a') as txt:
            txt.write(SEDpath[-13:-4]+'   ')
            for i in range(len(photometric_data)):
                txt.write('{:}   '.format(photometric_data[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        f = open(output_path+'lick_indices_'+'Z_'+SEDpath[-24:-18]\
                  +'.txt','r+')
        lines = f.readlines() # read old content
        f.seek(0) # go back to the beginning of the file
        f.write('# tau [Gyrs] '+str(lick_idx)+ D4000) 
        f.close()        
    
        f = open(output_path+'photometry_'+'Z_'+SEDpath[-24:-18]\
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
            
            if (len(photo_bands)!=0)&(len(lick_idx)!=0):
                
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
        
        
            # Writing output files
            # TODO: Modify the way of assigning the file names. Currently hardcoded.
            with open(output_path+'lick_indices_'+'Z_'+SEDpath[-24:-18]\
                      +'.txt', '+a') as txt:
                txt.write(SEDpath[-13:-4]+'   ')
                txt.write('   {:}   '.format(A_v_i))
                for i in range(len(lick_data)):
                    txt.write('   {:}   '.format(lick_data[i]))
                txt.write('\n') 
            txt.close
            del txt
            
            with open(output_path+'photometry_'+'Z_'+SEDpath[-24:-18]\
                     +'.txt', '+a') as txt:
                txt.write(SEDpath[-13:-4]+'   ')
                txt.write('   {:}   '.format(A_v_i))
                for i in range(len(photometric_data)):
                    txt.write('{:}   '.format(photometric_data[i]))
                txt.write('\n') 
            txt.close
            del txt
            
#        f = open(output_path+'lick_indices_'+'Z_'+SEDpath[-24:-18]\
#                  +'.txt','r+')
#        
#        f.write('# tau [Gyrs] --- E(B-V) ('+ext_law+')'+ 
#                '---'+str(lick_idx)+D4000+'\n') 
#        f.close()        
#    
#        f = open(output_path+'photometry_'+'Z_'+SEDpath[-24:-18]\
#                  +'.txt','r+')
#        
#        f.write('# tau [Gyrs] --- E(B-V) ('+ext_law+')'+ 
#                '---'+str(photo_bands)+'\n') 
#        f.close()                
    
    
    
    
    
    
    
    
    
