#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:37:22 2019

@author: pablo
"""

import numpy as np
import os
from matplotlib import pyplot as plt

class Lick_index(object):
    
    def __init__(self, **kwargs):
        
        flux = kwargs['flux']
        flux_err = kwargs['flux_err']
        lamb = kwargs['lamb']
        lick_index_name = kwargs['lick_index_name']
        
        # lambda must me expressed in AA
        self.blue_band, self.red_band, self.index_band, self.index_units = \
                                Lick_index.select_index_bands(lick_index_name) 
        
        self.lamb, self.flux, self.flux_err = \
                Lick_index.increase_resolution_pts(self, flux, flux_err, lamb)
    
        Lick_index.compute_pseudocont(self)
        Lick_index.compute_Lick(self)
            
    def increase_resolution_pts(self, flux, flux_err, lamb, resolution=1000):        
        
        new_lamb = np.linspace(self.blue_band[0],
                                     self.red_band[-1],
                                     resolution)
        new_flux = np.interp(new_lamb, lamb, flux)
        
        new_flux_err = np.interp(new_lamb, lamb, flux_err)
        
        return new_lamb, new_flux, new_flux_err
         
    def compute_pseudocont(self):
        
        # Take all points within the blue band: lambda<lambda_blue_2     
        left_lamb_pos = np.where(self.lamb<=self.blue_band[-1])[0]           
        mean_left_flux = np.nanmean(self.flux[left_lamb_pos])
        lamb_left = (self.blue_band[0]+self.blue_band[-1])/2
#        self.blue_error =  [np.nanstd(self.flux[left_lamb_pos]),
#                            len(left_lamb_pos)]
        delta_Cb = np.sqrt(np.nanmean(self.flux_err[left_lamb_pos]**2))
        
        # Take all points within the red band: lambda>lambda_red_1
        right_lamb_pos = np.where(self.lamb>=self.red_band[0])[0]    
        mean_right_flux = np.nanmean(self.flux[right_lamb_pos])    
        lamb_right = (self.red_band[0]+self.red_band[-1])/2
#        self.red_error =  [np.nanstd(self.flux[right_lamb_pos]), 
#                           len(right_lamb_pos)]
        delta_Cr = np.sqrt(np.nanmean(self.flux_err[right_lamb_pos]**2))
        
        # Fit to a straight line        
#        lamb_line = (lamb_left, lamb_right)
#        A = np.vstack([lamb_line, np.ones_like(lamb_line)]).T        
#        m, c = np.linalg.lstsq(A, [mean_left_flux, mean_right_flux])[0]
        
        delta_lamb = lamb_right-lamb_left
                
        
#        self.pseudocont = lambda lamb: c+m*lamb
        self.pseudocont = lambda lamb: mean_left_flux*((lamb_right-lamb)/delta_lamb)+\
        mean_right_flux*(lamb-lamb_left)/delta_lamb        
            
        self.pseudocont_err = lambda lamb: delta_Cb*\
        ((lamb_right-lamb)/delta_lamb)+\
        delta_Cr*(lamb-lamb_left)/delta_lamb        
            
    def compute_Lick(self):
        central_lamb_pts = np.where((self.lamb>=self.index_band[0])&(
                self.lamb<=self.index_band[-1]))[0]
        
        central_lamb = self.lamb[central_lamb_pts]
        
        central_flux = self.flux[central_lamb_pts]
        central_flux_err = self.flux_err[central_lamb_pts]
        
        central_pseudocont = self.pseudocont(central_lamb)
        central_pseudocont_err = self.pseudocont_err(central_lamb)
        
        central_lamb_int = np.insert(central_lamb, 0,
                                     2*central_lamb[0]-central_lamb[1])                                           
        delta_lamb  = np.ediff1d(central_lamb_int)
        
#        self.lick_index = np.trapz(1-central_flux/central_pseudocont, 
#                                   central_lamb)
        
        if self.index_units =='A':
            self.lick_index = np.nansum((1-central_flux/central_pseudocont)
            *delta_lamb
                )
        
        
            self.lick_index_err = np.nansum(
                (delta_lamb*central_flux_err/central_pseudocont)**2) +\
                np.nansum(
                (delta_lamb*central_pseudocont_err*central_flux/
                                       central_pseudocont**2)**2)
        
            self.lick_index_err = np.sqrt(self.lick_index_err)
                                       
        elif self.index_units=='mag':                                       
            self.lick_index = 1/(self.index_band[1]-self.index_band[0])*\
                        np.nansum(central_flux/central_pseudocont*delta_lamb)
            self.lick_index = -2.5*np.log10(self.lick_index)                        
            
            self.lick_index_err = np.nansum(
                (delta_lamb*central_flux_err/central_pseudocont)**2) +\
                np.nansum(
                (delta_lamb*central_pseudocont_err*central_flux/
                                       central_pseudocont**2)**2)
        
            self.lick_index_err = np.sqrt(self.lick_index_err)
#        self.lick_index_err = np.sqrt(
#                self.blue_error[0]**2/(self.blue_error[1]-1)\
#                +self.red_error[0]**2/(self.red_error[1]-1)
#                )
                    
        
    def select_index_bands(name):
        list_path = os.path.join(os.path.dirname(__file__), 'lick_list.txt')
        
        lick_list = np.loadtxt(list_path, usecols=0, dtype=str)
        lick_units = np.loadtxt(list_path, usecols=-1, dtype=str)
        
        lick_units = dict(zip(lick_list,lick_units))        
        index_band =  np.loadtxt(list_path, usecols=(1, 2))
        index_band = dict(zip(lick_list,index_band))
        blue_band = np.loadtxt(list_path, usecols=(3, 4))
        blue_band = dict(zip(lick_list,blue_band))
        red_band = np.loadtxt(list_path, usecols=(5, 6))
        red_band = dict(zip(lick_list, red_band))
        

        return blue_band[name],red_band[name],index_band[name],lick_units[name]
    
    def plot_index(self, data=False, folder=''):
        plt.switch_backend('agg')
        if data!=False:
            plt.figure(figsize=(8,7))
            plt.title(data)        
            plt.plot(self.lamb, self.flux, '.', c='k')
            plt.plot(self.lamb, self.flux,  '-', c='k', label='flux')
            plt.plot(self.lamb, self.flux_err, label=r'1 $\sigma$ flux')
            plt.plot(self.lamb,self.pseudocont_err(self.lamb), 
                     label='pseudocont_err')
            plt.plot(self.lamb,self.pseudocont(self.lamb), c='r', 
                     label='pseudocont')
            plt.vlines(self.index_band[0], np.nanmin(self.flux),
                       np.nanmax(self.flux), color='b')
            plt.vlines(self.index_band[-1], np.nanmin(self.flux),
                       np.nanmax(self.flux), color='b')            
            plt.xlabel(r'Wavelength [\AA]')
            plt.text(self.lamb[20], np.nanmax(self.flux_err)*1.5, 'L_idx= '+   
                str(self.lick_index)+'\n L_idx_err= '+ str(self.lick_index_err))
            plt.legend()
            if folder!='':
                plt.savefig(folder, bbox_inches='tight')
                plt.close()
        else:
            plt.figure(figsize=(8,7))
            plt.plot(self.lamb, self.flux, '.', c='k')
            plt.plot(self.lamb, self.flux,  '-', c='k', label='flux')
            plt.plot(self.lamb, self.flux_err, label=r'1 $\sigma$ flux')
            plt.plot(self.lamb,self.pseudocont_err(self.lamb), 
                     label='pseudocont_err')
            plt.plot(self.lamb,self.pseudocont(self.lamb), c='r', 
                     label='pseudocont')
            plt.vlines(self.index_band[0], np.nanmin(self.flux), 
                       np.nanmax(self.flux), color='b')
            plt.vlines(self.index_band[-1], np.nanmin(self.flux), 
                       np.nanmax(self.flux), color='b')            
            plt.xlabel(r'Wavelength [\AA]')
            plt.text(self.lamb[20], np.nanmax(self.flux_err)*1.5, 'L_idx= '+ \
                str(self.lick_index)+'\n L_idx_err= '+ str(self.lick_index_err))
            plt.legend()
            if folder!='':
                plt.savefig(folder, bbox_inches='tight')
                plt.close()
                
class BalmerBreak(object):
    """This class computes the Balmer Break at 4000 AA 
     by compution the ratio between the flux at F[4050-4250]/F[3750-3950]
     based on Bruzual 89.
     """
    def __init__(self, flux, lamb):
        # lambda must me expressed in AA        
        self.lamb, self.flux = BalmerBreak.increase_resolution_pts(self,
                                                                  flux,
                                                                  lamb)
        BalmerBreak.compute_D4000(self)
                    
    def increase_resolution_pts(self, flux, lamb, resolution=1000):        
        new_lamb = np.linspace(3750, 4250, resolution)
        new_flux = np.interp(new_lamb, lamb, flux)
        return new_lamb, new_flux
         
    def compute_D4000(self):
        # Take all points within the blue band: lambda<lambda_blue_2     
        left_lamb_pos = np.where(self.lamb<=3950)[0]           
        mean_left_flux = np.nanmedian(self.flux[left_lamb_pos])
        # Take all points within the red band: lambda>lambda_red_1
        right_lamb_pos = np.where(self.lamb>=4050)[0]    
        mean_right_flux = np.nanmedian(self.flux[right_lamb_pos])    

        delta_blue = np.nanstd(self.flux[left_lamb_pos])**2/(
                len(self.flux[left_lamb_pos])-1)
        delta_red = np.nanstd(self.flux[right_lamb_pos])**2/(
                len(self.flux[right_lamb_pos])-1)
        
        self.D4000 = mean_right_flux/mean_left_flux
        
        self.sigma_D4000 = np.sqrt(self.D4000**2*
                    (delta_blue/mean_left_flux**2+delta_red/mean_right_flux**2))
        

        