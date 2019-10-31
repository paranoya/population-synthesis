#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:56:46 2018

@author: pablo
"""

import numpy as np

import os
import units
#import scipy as sp
from scipy import interpolate



# =============================================================================
class Filter(object):
# =============================================================================

    def __init__(self, **kwargs):
        """This class provides a filter (SDSS, WISE photometry) with the same 
        number of points as the given wavelength array.
        
        The wavelength UNITS are by default expressed in AA"""       
        self.wavelength = kwargs['wavelength']
        filter_name =kwargs['filter_name']
        
        if self.wavelength[5]>self.wavelength[6]:
            raise NameError('Wavelength array must be crescent') 
            
            
        self.filter_resp, self.wl_filter = Filter.get_filt(filter_name)
        self.filter = Filter.new_filter(self.wl_filter, 
                                        self.filter_resp, 
                                        self.wavelength)    
    def get_filt(filter_name):
       
        absolute_path = '/home/pablo/population-synthesis/dust_grid/products/Filters/'
        filters_path = {'u':absolute_path+'SDSS/u.dat',
                       'g':absolute_path+'SDSS/g.dat',
                       'r':absolute_path+'SDSS/r.dat',
                       'i':absolute_path+'SDSS/i.dat',
                       'z':absolute_path+'SDSS/z.dat', 
                       'W1':absolute_path+'WISE/W1.dat',
                       'W2':absolute_path+'WISE/W2.dat',
                       'W3':absolute_path+'WISE/W3.dat', 
                       'W4':absolute_path+'WISE/W4.dat'} 
       
       
        filt=np.loadtxt(filters_path[filter_name], usecols=1)
        w_l=np.loadtxt(filters_path[filter_name], usecols=0)
        
        return filt, w_l


    
    def new_filter( wl, fil, new_wl,*name, save=False):
        """ This function recieve the filter response and wavelength extension in order to interpolate it to a new set
         wavelengths.  First, it is checked if the filter starts or ends on the edges of the data, 
         if this occurs an array of zeros is added to limit the effective area. 
         Then, the filter response is differenciated seeking the limits of the curve to prevent wrong extrapolation. """
        
        f=interpolate.interp1d( wl, fil , fill_value= 'extrapolate' )
        
        new_filt=f(new_wl)
        
        bad_filter = False
        
        if  len(np.where(fil[0:5]>0.05)[0]):                           
            fil = np.concatenate((np.zeros(100),fil))
            bad_filter = True
        elif len(np.where(fil[-5:-1]>0.05)[0]):
            fil = np.concatenate((fil, np.zeros(100)))
            bad_filter = True
                                    
        
        band_init_pos = np.where(fil>0.01)[0][0]
        band_end_pos = np.where(fil[::]>0.01)[0][0]
        
        wl_init_pos = wl[band_init_pos]
        wl_end_pos = wl[-band_end_pos]
 
        new_band_init_pos = (np.abs(new_wl-wl_init_pos)).argmin()    
        new_band_end_pos = (np.abs(new_wl-wl_end_pos)).argmin()
        
        # To smooth the limits of the band, first the band width is computed (number of points inside) and then a 
        # called 'tails' to avoid erase any value close to te edge. If the filter starts at one corner of the distribution
        # obviously band_width_pos > new_band_init_pos, so the 'tails' could introduce negative positions. In order to avoid
        # this effect it is better to use the own initial position to delimitate the 'tail' of the band. But also, another 
        # problem is the possible lack of points and then the tail would be underestimated. For this reason, is estimated
        # the number of points out of the new distribution and the tail is enlarged proportionally.
        
        band_width_pos =  new_band_end_pos - new_band_init_pos
        
        band_tails_right_pos = int(band_width_pos*0.1)
        
        band_tails_left_pos  = band_tails_right_pos
        
        if band_width_pos>new_band_init_pos:
            missing_points = 0            
            if new_band_init_pos==0:
                delta_wl =  np.mean(np.ediff1d(new_wl[0:100]))
                missing_points = (new_wl[0] - wl_init_pos )/delta_wl
                                
            band_tails_left_pos = int(new_band_init_pos*0.1)
            band_tails_right_pos = int(band_width_pos*0.1)+int(missing_points*0.1)
            
        elif band_width_pos > len(new_wl)-new_band_end_pos:
            missing_points = 0            
            if new_band_end_pos==(len(new_wl)-1):
                delta_wl =  np.mean(np.ediff1d(new_wl[-100:-1]))
                missing_points = (wl_end_pos -new_wl[0] )/delta_wl
                                
            band_tails_left_pos = int(band_width_pos*0.1)
            band_tails_right_pos = int((len(new_wl)-new_band_end_pos)*0.1)+int(missing_points*0.1)
            
           
        new_filt[0:new_band_init_pos-band_tails_left_pos] = np.clip(new_filt[0:(new_band_init_pos-band_tails_left_pos)],0,0)
        new_filt[(new_band_end_pos+band_tails_right_pos):-1] = np.clip(new_filt[(new_band_end_pos+band_tails_right_pos):-1],0,0)
        
        new_filt[-1]=0     # Sometimes it is the only point which gives problems
        
        # Furthermore, the worst case is when the original filter also starst at one corner of the distrib, so probably wrong
        # values appear close to the real curve. More drastically, all the sorrounding points are set to zero.
        
        if bad_filter == True:
            new_filt[new_band_end_pos:-1]=0
            new_filt[0:new_band_init_pos]=0
        
        if save==True:
            new_filt_zip=zip(new_wl,new_filt)
            
            with open('Filters/'+str(name)+'.txt', 'w' ) as f:
                for i,j in new_filt_zip:
                    f.write('{:.4} {:.4}\n'.format(i,j))
            print('Filter'+str(name)+'saved succesfully ')        
        
        return new_filt    
    
    
    def square_filt(wl,l_i,l_e):
         
         s_filt=np.ones(len(wl))
         
         for i,j in enumerate(wl):
             if j<l_i:
                 
                 s_filt[i]=0
            
             elif j>l_i and j<l_e:
                 
                 s_filt[i]=1
            
             elif j>l_e:
                 
                 s_filt[i]=0
         return s_filt
         
         
         
# =============================================================================
class magnitude(Filter):         
# =============================================================================
    """This module computes the photmetric magnitude on a given band 
    for a given flux with UNITS expressed in erg/s.s"""
             
    def __init__(self, absolute=False, **kwargs):
        
        Filter.__init__(self, **kwargs)
        self.nu = units.c/( self.wavelength*1e-10 )
        self.flux = kwargs['flux']
        self.absolute = absolute
        photometric_system = kwargs['photometric_system']
        
        if photometric_system=='AB': 
            self.magnitude = magnitude.AB(self)
        if photometric_system=='Vega':
            self.magnitude = magnitude.Vega(self)
    
    def AB(self):   #photometric system  used with SDSS filters
        """ This function computes the magnitude in the AB system of a given spectrum. The spectrum units must be in erg/s for absolute
         magnitude computation or in erg/s/cm2 for apparent magnitude. """
 
        if self.absolute==True: 
            self.flux = self.flux/(4*np.pi* (10*units.pc/units.cm)**2)   # flux at 10 pc.
        
        
        diff_nu = - np.ediff1d(np.insert(self.nu, 0, 2*self.nu[0]-self.nu[1]))
        
        integral_flux = np.nansum((self.flux/self.nu * self.filter * diff_nu) )        
        
        integral_R = np.nansum(self.filter*diff_nu)
        
        mag =-2.5*np.log10(integral_flux/integral_R) - 48.60
        return mag
         
    def Vega(self): #photometric system  used usually with Jonhson/Bessel/Coussin filters
        
        diff_wl = np.ediff1d(np.insert((self.wavelength),0,0))
            
        wl_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=0)
        diff_wl_vega = np.ediff1d(np.insert(wl_vega,0,0))
        flux_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=2)
        
        if self.absolute == True:
            flux_vega=( flux_vega * 4*np.pi*(25.30*units.ly/units.cm)**2 ) / (4*np.pi* (10*units.pc/units.cm)**2)
            self.flux = self.flux/(4*np.pi* (10*units.pc/units.cm)**2)   #flux at 10 pc
            
            
        vega_filter=Filter.new_filter(self.wl_filter, self.filter_resp , wl_vega)
        integral_flux_vega=np.nansum(flux_vega * vega_filter * diff_wl_vega)
        integral_flux = np.nansum(self.flux * self.filter * diff_wl ) 
        
        m=-2.5* np.log10(integral_flux/integral_flux_vega) +0.58
         
        return m
    
