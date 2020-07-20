#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:31:13 2019

@author: pablo
"""

import os
from astropy.io import fits
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units


class read_spectra(object):
    """ This class recieves a spectrum path as input. It returns the flux,
    restframe wavelenghts and flux errors"""
    
    def __init__(self, path, survey):
        
        self.hdul = fits.open(path)
        read_method ={
                'SDSS':read_spectra.read_SDSS,
                'GAMA':read_spectra.read_GAMA,
                'MANGA':read_spectra.read_MANGA
                    }        
        read_method[survey](self)

#        try:
#            read_method[survey](self)
#        except:
#            raise NameError('File not found!')
    
    def read_SDSS(self):
        
        self.flux = self.hdul[1].data['flux']
        inverse_variance = self.hdul[1].data['ivar']
        self.sigma= np.sqrt(1/inverse_variance)
        self.redshift = self.hdul[2].data['Z']

        self.wavelength =  10**(self.hdul[1].data['loglam'])
        self.wavelength = self.wavelength/(1+self.redshift)
        
        self.flux = self.flux*self.wavelength*1e-17 # erg/s/cm2
        self.sigma = self.sigma*self.wavelength*1e-17 # erg/s/cm2 
        
    def read_GAMA(self):
        
        data = self.hdul[0].data
        lambda_c = self.hdul[0].header['CRVAL1']
        delta_lambda = self.hdul[0].header['CD1_1']        
        self.redshift = self.hdul[0].header['Z']
        
        self.hdul.close()    
        self.flux = data[0]
        self.sigma = data[1]
        
        self.wavelength = range(-len(self.flux)//2, len(self.flux)//2)
        self.wavelength =lambda_c + delta_lambda*np.array(self.wavelength)
        self.wavelength = self.wavelength /(1+self.redshift)
        
        self.flux = self.flux*self.wavelength*1e-17 # erg/s/cm2
        self.sigma = self.sigma*self.wavelength*1e-17 # erg/s/cm2    
        
    def read_MANGA(self):
        
        self.flux = self.hdul[1].data        
        self.inverse_variance = self.hdul[2].data                
        
        self.sigma = np.sqrt(1/self.inverse_variance)        
        self.wavelength = self.hdul[6].data
        self.manga_id = self.hdul[0].header['MANGAID']
        
        self.flux = self.flux*self.wavelength[:, np.newaxis, np.newaxis]*1e-17 # erg/s/cm2
        self.sigma = self.sigma*self.wavelength[:, np.newaxis, np.newaxis]*1e-17 # erg/s/cm2    
        MANGA_LIST =os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'MANGA', 'drpall-v2_4_3.fits')
        print(MANGA_LIST)
        hdul = fits.open(MANGA_LIST)    
        print(1)
        manga_id_list = hdul[1].data['mangaid']
        redshift_list = hdul[1].data['Z']
        
        self.manga_pos = np.where(manga_id_list==self.manga_id)[0]
        self.redshift = redshift_list[self.manga_pos]

        self.wavelength = self.wavelength /(1+self.redshift)    
        
    def get_keys(self):
        pass
    
    def get_SDSSinfo(self):
        
        self.specobjid = self.hdul[2].data['bestobjid']
        self.plate = self.hdul[2].data['plate']
        self.mjd = self.hdul[2].data['MJD']
        self.fiberid = self.hdul[2].data['fiberid']
        self.fluxobjid = self.hdul[2].data['fluxobjid']
        
    def flux_to_luminosity(self):
        lum_distance = cosmo.luminosity_distance(self.redshift) ## pc
        lum_distance = lum_distance.to(units.cm)
        self.luminosity = self.flux*4*np.pi*(lum_distance.value)**2 ## erg/s
        

        
        