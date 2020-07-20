#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:30:45 2019

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
import astropy.io.fits as fits

# =============================================================================
# Select model SED folder
# =============================================================================

SEDfolder = '../../GAMA_data/spectra_files/*.fit'


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

output_path = '../../GAMA_data/products/test/'
                
                

# =============================================================================
# =============================================================================
# =============================================================================
#%% Computation without dust models
# =============================================================================
# =============================================================================
# =============================================================================
                
if dust_grid ==False:                
    for i_element in range(10):
#    for i_element in range(5):
        print('\n {:.2}'.format(i_element/len(SEDpaths)*100)+' % complete -->' \
              + SEDpaths[i_element])
        
        hdul = fits.open(SEDpaths[i_element])     
        data = hdul[0].data
        cataid = hdul[0].header['CATAID']
        SN = hdul[0].header['SN']
        lambda_c = hdul[0].header['CRVAL1']
        lambda_c_pos = int(hdul[0].header['CRPIX1'])
        delta_lambda = hdul[0].header['CD1_1']
        
        z_redshift = hdul[0].header['Z']
        
        hdul.close()
        del hdul
        
        flux = data[0]
        flux_err = data[1]
        
        wavelength = range(-len(flux)//2, len(flux)//2)
        wavelength =lambda_c + delta_lambda*np.array(wavelength)
        wavelength = wavelength /(1+z_redshift)
        
        
        flux = flux*wavelength*1e-17 # erg/s/cm2
        flux_err = flux_err*wavelength*1e-17 # erg/s/cm2
        
        if (len(photo_bands)!=0)&(len(lick_idx)!=0):
            
            photometric_data= []
            lick_data = []
            lick_err = []
            
            if len(lick_idx)>len(photo_bands):
                primary_loop_length = len(lick_idx)
               
                for element in range(primary_loop_length):
                
                    l_idx  = lick_indices.Lick_index(
                            flux=flux, 
                            flux_err=flux_err,
                            lamb=wavelength, 
                            lick_index_name= lick_idx[element])
#                    if (l_idx.lick_index_err/l_idx.lick_index>0.4)|(l_idx.lick_index==np.nan):
                    l_idx.plot_index(
        data='CATAID: '+str(cataid)+', SN: '+str(SN)+', lindex: '+lick_idx[element],
        folder=output_path+str(cataid)+'_'+lick_idx[element]+'.png'
        )
                        
                    lick_data.append(l_idx.lick_index)
                    lick_err.append(l_idx.lick_index_err)
                    if element<len(photo_bands):
                        mag = photometry.magnitude(absolute=False, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                        photometric_data.append(mag.magnitude)
                if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    lick_err.append(bbreak.sigma_D4000)
            else:
                primary_loop_length = len(photo_bands)
                
                
                for element in range(primary_loop_length):
                
                    mag = photometry.magnitude(absolute=False, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                    photometric_data.append(mag.magnitude)
                    
                    if element<len(lick_idx):
                        
                        l_idx  = lick_indices.Lick_index(
                            flux=flux, 
                            flux_err=flux_err,
                            lamb=wavelength, 
                            lick_index_name= lick_idx[element])
                        
#                        if (l_idx.lick_index_err/l_idx.lick_index>0.4)|(l_idx.lick_index==np.nan):
#                            l_idx.plot_index(
#        data='CATAID: '+str(cataid)+', SN: '+str(SN)+', lindex: '+lick_idx[element],
#        folder=output_path+'lick_index_failures/'+str(cataid)+'_'+lick_idx[element]+'.png'
#        )
                        lick_data.append(l_idx.lick_index)
                        lick_err.append(l_idx.lick_index_err)
                        
                if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    lick_err.append(bbreak.sigma_D4000)
        elif photo_bands==False:
            
            lick_data = []
            lick_err = []
            for element in range(len(lick_idx)):
                
                    l_idx  = lick_indices.Lick_index(
                            flux=flux, 
                            flux_err=flux_err,
                            lamb=wavelength, 
                            lick_index_name= lick_idx[element])
#                    if (l_idx.lick_index_err/l_idx.lick_index>0.4)|(l_idx.lick_index==np.nan):
#                        l_idx.plot_index(
#        data='CATAID: '+str(cataid)+', SN: '+str(SN)+', lindex: '+lick_idx[element],
#        folder=output_path+'/lick_index_failures/'+str(cataid)+'_'+lick_idx[element]+'.png'
#        )
                    lick_data.append(l_idx.lick_index)
                    lick_err.append(l_idx.lick_index_err)
                    
            if balmer_break==True:
                    bbreak = lick_indices.BalmerBreak(flux, wavelength)
                    lick_data.append(bbreak.D4000)
                    lick_err.append(bbreak.sigma_D4000)
        else:
            
            photometric_data= []
            
            for element in range(len(photo_bands)):
                
                    mag = photometry.magnitude(absolute=False, 
                                               filter_name=photo_bands[element], 
                                               wavelength=wavelength, flux=flux,
                                               photometric_system=photo_system)
                    photometric_data.append(mag.magnitude)
    
    
        # Writing output files
        # TODO: Modify the way of assigning the file names. Currently hardcoded.
        # TODO: Adaptative file content (for cases with no photometry/lick indices)
        with open(output_path+'GAMA_lick_indices.txt', '+a') as txt:
            txt.write('{:}  '.format(cataid))
            for i in range(len(lick_data)):
                txt.write('  {:}  {:}  '.format(lick_data[i], lick_err[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        with open(output_path+'GAMA_photometry.txt', '+a') as txt:
            txt.write('{:}  '.format(cataid))
            for i in range(len(photometric_data)):
                txt.write('{:}   '.format(photometric_data[i]))
            txt.write('\n') 
        txt.close
        del txt
        
        del l_idx
        del mag
        del lick_data
        del lick_err
        del photometric_data
        
    f = open(output_path+'GAMA_lick_indices.txt','+a')    
    f.write('# CATAID '+str(lick_idx)+ D4000) 
    f.close()        

    f = open(output_path+'GAMA_photometry.txt','+a')    
    f.write('# CATAID '+str(photo_bands)) 
    f.close()                
        
# =============================================================================
# =============================================================================
# =============================================================================
