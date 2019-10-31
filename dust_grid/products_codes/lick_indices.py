#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:37:22 2019

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt

class Lick_index(object):
    
    def __init__(self, **kwargs):
        
        flux = kwargs['flux']
        flux_err = kwargs['flux_err']
        lamb = kwargs['lamb']
        lick_index_name = kwargs['lick_index_name']
        
        # lambda must me expressed in AA
        self.blue_band, self.red_band, self.index_band = Lick_index.select_index_bands(lick_index_name) 
        self.lamb, self.flux, self.flux_err = Lick_index.increase_resolution_pts(self,
                                                                  flux, 
                                                                  flux_err,
                                                                  lamb)
    
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
        self.lick_index = np.nansum((1-central_flux/central_pseudocont)
        *delta_lamb
                )
        
        
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
        blue_band = {'HdeltaA': [4041.600, 4079.750], 
                     'HdeltaF': [4057.250, 4088.500],
                     'Ca4227' : [4211.000, 4219.750],
                     'HgammaA': [4283.500, 4319.750], 
                     'HgammaF': [4283.500, 4319.750],
                     'Fe4383' : [4359.125, 4370.375],
                     'Ca4455' : [4445.875, 4454.625],
                     'Hbeta': [4827.875, 4847.875],
                     'Mg1' : [4895.125, 4957.625],
                     'Mgb' : [5142.625, 5161.375],
                     'Fe5270' : [5233.150, 5248.150],
                     'Fe5335': [5306.625, 5317.875],
                     'DTT_CaII8498':[8447.5, 8462.5]
                     }
        
        index_band = {'HdeltaA': [4083.500, 4122.250], 
                     'HdeltaF': [4091.000, 4112.250],
                     'Ca4227' : [4222.250, 4234.750],
                     'HgammaA': [4319.750, 4363.500], 
                     'HgammaF': [4331.250, 4352.250],
                     'Fe4383' : [4369.125, 4420.375],
                     'Ca4455' : [4452.125, 4474.625],
                     'Hbeta': [4847.875, 4876.625], 
                     'Mg1' : [5069.125, 5134.125],
                     'Mgb' : [5160.125, 5192.625],
                     'Fe5270' : [5245.650, 5285.650],
                     'Fe5335': [5314.125, 5354.125],
                     'DTT_CaII8498':[8483.0, 8513.0]
                    }   
        
        red_band = {'HdeltaA': [4128.500, 4161.000], 
                     'HdeltaF': [4114.750, 4137.250],
                     'Ca4227' : [4241.000, 4251.000],
                     'HgammaA': [4367.250, 4419.750], 
                     'HgammaF': [4354.750, 4384.750],
                     'Fe4383' : [4442.875, 4455.375],
                     'Ca4455' : [4477.125, 4492.125],
                     'Hbeta': [4876.625, 4891.625], 
                     'Mg1' : [5301.125, 5366.125],
                     'Mgb' : [5191.375, 5206.375],
                     'Fe5270' : [5285.650, 5318.150],
                     'Fe5335': [5355.375, 5365.375],
                     'DTT_CaII8498':[8842.5, 8857.5]
                    }
         
        
        # return (lambda_low, lambda_up) for each band
        return blue_band[name], red_band[name], index_band[name]
    
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
            plt.text(self.lamb[20], np.nanmax(self.flux_err)*1.5, 'L_idx= '+ \
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
        
        
# =============================================================================
#         SDSS LICK DEFINITIONS
# =============================================================================
#'Lick_CN1':[4143.375, 4178.375, 4081.375, 4118.875, 4245.375, 4285.375],
#'Lick_CN2':[4143.375, 4178.375, 4085.125, 4097.625,	4245.375, 4285.375],
#'Lick_Ca4227':[4223.500, 4236.000 ,4212.250 ,4221.000 ,4242.250 ,4252.250],
#'Lick_G4300':[4282.625,	4317.625, 4267.625, 4283.875, 4320.125, 4333.375],
#'Lick_Fe4383':[4370.375, 4421.625, 4360.375, 4371.625,	4444.125, 4456.625],
#'Lick_Ca4455':[4453.375, 4475.875,	4447.125, 4455.875,	4478.375, 4493.375],
#'Lick_Fe4531':[4515.500, 4560.500, 	4505.500, 4515.500, 4561.750, 4580.500],
#'Lick_C4668':[4635.250,	4721.500, 4612.750, 4631.500, 4744.000, 4757.750],
#'Lick_Hb':[4848.875, 4877.625,	4828.875, 	4848.875, 	4877.625, 	4892.625],
#'Lick_Fe5015':[4979.000, 5055.250, 	4947.750, 4979.000, 5055.250, 5066.500],
#'Lick_Mg1':[5070.375, 5135.375, 4896.375, 4958.875, 5302.375, 5367.375],
#'Lick_Mg2':[5155.375, 5197.875, 4896.375, 4958.875, 5302.375, 5367.375],
#'Lick_Mgb':[5143.875, 5162.625, 5161.375, 5193.875, 5192.625, 5207.625],
#Lick_Fe527 	5247.375 	5287.375 	5234.875 	5249.875 	5287.375 	5319.875
#Lick_Fe5335 	5314.125 	5354.125 	5306.625 	5317.875 	5355.375 	5365.375
#Lick_Fe5406 	5390.250 	5417.750 	5379.000 	5390.250 	5417.750 	5427.750
#Lick_Fe5709 	5698.375 	5722.125 	5674.625 	5698.375 	5724.625 	5738.375
#Lick_Fe5782 	5778.375 	5798.375 	5767.125 	5777.125 	5799.625 	5813.375
#Lick_NaD 	5878.625 	5911.125 	5862.375 	5877.375 	5923.875 	5949.875
#Lick_TiO1 	5938.875 	5995.875 	5818.375 	5850.875 	6040.375 	6105.375
#Lick_TiO2 	6191.375 	6273.875 	6068.375 	6143.375 	6374.375 	6416.875
#B&H_CNB 	3810.0 	3910.0 	3785.0 	3810.0 	3910.0 	3925.0
#B&H_H=K 	3925.0 	3995.0 	3910.0 	3925.0 	3995.0 	4010.0
#B&H_CaI 	4215.0 	4245.0 	4200.0 	4215.0 	4245.0 	4260.0
#B&H_G 	4285.0 	4315.0 	4275.0 	4285.0 	4315.0 	4325.0
#B&H_CaI 	4215.0 	4245.0 	4200.0 	4215.0 	4245.0 	4260.0
#B&H_G 	4285.0 	4315.0 	4275.0 	4285.0 	4315.0 	4325.0
#B&H_Hb 	4830.0 	4890.0 	4800.0 	4830.0 	4890.0 	4920.0
#B&H_MgG 	5150.0 	5195.0 	5125.0 	5150.0 	5195.0 	5220.0
#B&H_MH 	4940.0 	5350.0 	4740.0 	4940.0 	5350.0 	5550.0
#B&H_FC 	5250.0 	5280.0 	5225.0 	5250.0 	5280.0 	5305.0
#B&H_NaD 	5865.0 	5920.0 	5835.0 	5865.0 	5920.0 	5950.0
#DTT_CaII8498 	8483.0 	8513.0 	8447.5 	8462.5 	8842.5 	8857.5
#DTT_CaII8542 	8527.0 	8557.0 	8447.5 	8462. 	8842.5 	8857.5
#DTT_CaII8662 	8467.0 	8677.0 	8447.5 	8462.5 	8842.5 	8857.5
#DTT_MgI8807 	8799.5 	8814.5 	8775.0 	8787.0 	8845.0 	8855.0        
        
        