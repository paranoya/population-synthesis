# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:20:11 2019

@author: corch
"""

import numpy as np


class DustModelGrid(object):
    
    def __init__(self, flux, wavelength, ext_law_name, dust_dimension):
        """ This class object takes as arguments the unreddened flux, 
        the wavelenght corresponing to each flux point, the desired extinction
        law to apply 
        and the number of different models that want to be returned
        """
        
        self.flux_clean = flux
        self.wavelength = wavelength        
                
        DustModelGrid.select_ext_law(self, ext_law_name)
        extinction_grid = DustModelGrid.extinction_grid_builder(
                self, dust_dimension, ext_law_name)
        DustModelGrid.SEDgrid_generator(self, extinction_grid)
        
    def select_ext_law(self, name):
        """
        Possible extinction laws:
            - Cardelli 89
        """
        ext_laws = {'cardelli89': extinction_laws.Cardelli89_ext_law, 
                    'calzetti2000': extinction_laws.Calzetti2000_ext_law}
        
        self.ext_law = ext_laws[name]           
        
        
    def extinction_grid_builder(self, n_models, name):
        
        if name=='cardelli89':
            """ For each value o the extinction coefficient in the visual magnitude
            A_V, a different extinction curve will be produced for the whole 
            spectrum. The lower value must be 0 (where no extinction takes place)
            and an upper value chosen to be 5.
            """
            self.A_V = np.linspace(0, 3, n_models)
            
            A_lambda = self.ext_law(self.wavelength)
            
            A_lambda_grid = A_lambda[:, np.newaxis]*self.A_V[np.newaxis, :]
            
            return A_lambda_grid
        
        if name=='calzetti2000':
            """ For each value o the extinction coefficient in the visual magnitude
            A_V, a different extinction curve will be produced for the whole 
            spectrum. The lower value must be 0 (where no extinction takes place)
            and an upper value chosen to be 5.
            """
            self.colorexcess_B_V = np.linspace(0, 1, n_models)
            
            A_lambda = self.ext_law(self.wavelength)
            
            A_lambda_grid = A_lambda[:, np.newaxis]\
            *self.colorexcess_B_V[np.newaxis, :]
            
            return A_lambda_grid    
    
    def SEDgrid_generator(self, extinction_grid):
        """ This function produces the SED grid, where the 0th dimension 
        correspond to the wavelenght dimension and the 1st dimension corresponds
        with the reddening.
        """
        self.SED_grid = self.flux_clean[:, np.newaxis] * \
                                                    10**(-0.4*extinction_grid)
        
    
class extinction_laws():
    
    def Cardelli89_ext_law(_lambda, RV=3.1):
        """This function return the extinction law provided by Cardelli 1989.
        It is expressed in the form of A_lambda/A_V = f(lambda, RV).
        Typical (and default) values for the interestelar medium are Rv = 3.1
        """
        if _lambda[0]>_lambda[1]:
            raise NameError('Wavelength array must be crescent') 
            
        x = 1/_lambda
        #    Infrared Warning: this regime is valid for 0.3<=x<=1.1 um^-1, 
        # but is extrapolated 
        # for longer wavelengths  
        a_x_IR = 0.574*x[x<=1.1]**1.61
        
        b_x_IR = -0.527*x[x<=1.1]**1.61
        #   Optical/NIR
        y = x[(1.1<x)&(x<=3.3)]-1.82
        
        a_x_OP = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
                + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b_x_OP = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
                - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
        # UV
        F_a = np.zeros_like(x[(3.3<x)&(x<=8)])
        F_b = np.zeros_like(x[(3.3<x)&(x<=8)])
        
        UV_pts = np.where((5.9<=x[(3.3<x)&(x<=8)])&(x[(3.3<x)&(x<=8)]<=8))[0]
    
        F_a[UV_pts] = -0.04473*(x[(5.9<=x)&(x<=8)]-5.9)**2 \
            - 0.009779*(x[(5.9<=x)&(x<=8)]-5.9)**3
        F_b[UV_pts] = 0.2130*(x[(5.9<=x)&(x<=8)]-5.9)**2 \
            + 0.1207*(x[(5.9<=x)&(x<=8)]-5.9)**3
                        
        a_x_UV = 1.752 - 0.316*x[(3.3<x)&(x<=8)] - 0.104/(
                (x[(3.3<x)&(x<=8)]-4.67)**2 +0.341)   +F_a
        b_x_UV = -3.090 + 1.825*x[(3.3<x)&(x<=8)]  + 1.206/(
                (x[(3.3<x)&(x<=8)]-4.62)**2  +0.263)   +F_b
        # FUV Warning: this regime is valid for 8<x<10 um^-1, but is extrapolated 
        # for shorter wavelengths
        
        a_x_FUV = -1.073 - 0.628*(x[x>8]-8) + 0.137*(x[x>8]-8)**2 \
        - 0.07*(x[x>8]-8)**3
        b_x_FUV = 13.670 + 4.257*(x[x>8]-8) - 0.42*(x[x>8]-8)**2 \
        + 0.374*(x[x>8]-8)**3
            
        a_x = np.concatenate((a_x_FUV, a_x_UV, a_x_OP, a_x_IR))
        b_x = np.concatenate((b_x_FUV, b_x_UV, b_x_OP, b_x_IR))
        A_over_AV = a_x + b_x/RV 
        
        return A_over_AV                        
                                 
    def Calzetti2000_ext_law(_lambda, RV=3.1):
        """This function return the extinction law provided by Calzetti 2000.
        It is expressed in the form of k(lambda)/E(B-V) = f(lambda, RV).
        Typical (and default) values for the interestelar medium are Rv = 3.1
        """
        if _lambda[0]>_lambda[1]:
            raise NameError('Wavelength array must be crescent') 
            
        x = 1/_lambda
        print(x)
        #    Infrared Warning: this regime is valid for 0.45<=x<=1.58 um^-1, 
        # but is extrapolated 
        # for longer wavelengths  
        k_lambda_IR = 2.659*(-1.857 + 1.04*x[x<=1.58]) +RV
        
        k_lambda_OP = 2.659*(-2.156 + 1.509*x[x>1.58] - 0.198*x[x>1.58]**2
                             +0.011*x[x>1.58]**3) +RV
                             
        k_lambda = np.concatenate((k_lambda_OP, k_lambda_IR))
        
        return k_lambda                                    
                    
                    
        