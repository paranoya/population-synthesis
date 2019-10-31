#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:31:13 2019

@author: pablo
"""

from astropy.io import fits
import numpy as np



class read_spectra(object):
    """ This class recieves a spectrum path as input. It returns the flux,
    restframe wavelenghts and flux errors"""
    
    def __init__(self, path, survey):
        
        self.hdul = fits.open(path)

        read_method ={
                'SDSS':read_spectra.read_SDSS,
                'GAMA':read_spectra.read_GAMA
                    }        
        
        try:
            read_method[survey](self)
        except:
            raise NameError('File not found!')
    
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
        
        
    def get_keys(self):
        pass
    def get_info(self):
        pass
        
        
        
        