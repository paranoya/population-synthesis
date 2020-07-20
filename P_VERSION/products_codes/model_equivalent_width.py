import numpy as np 
import units

class model_equivalent_width(object):
    """
    This module provides a tool for estimating the nebular emission contribution
    to the balmer lines an computes the resultant equivalent width
    """
    T_e={ 0.0001:19950., 0.0004:15850., 0.004:10000., 0.008:7940.,
         0.02:6310., 0.05:3160.}        # T_e(Z) [K]        
    a_over_b={ 0.05:3.30, 0.02:3.05, 0.008:2.87, 0.004:2.87, 0.0004:2.76,
              0.0001:2.76 }                    
    
    
    def __init__(self, **kwargs):
        self.Z = kwargs['Z']
        
        self.flux = kwargs['flux']  ## input flux units == erg/s.
        # Later this quantity is divided by the wavelength erg/AA/s

        try:
            self.flux_err = kwargs['flux_err']
        except:
            self.flux_err = np.zeros_like(self.flux)

            
        self.wavelength = kwargs['wavelength'] # lambda must me expressed in AA
        self.line = kwargs['line']
        
        self.Te = self.T_e[self.Z]
        self.alf_B()
        self.j_B()
        self.compute_Q()
        self.compute_line_emission()
        self.compute_EW()
        

    def compute_Q(self):
        if (self.line=='hbeta')|(self.line=='halpha'):
            hydrogen_lim = 912 #AA
            wl_pts = np.where(self.wavelength<=hydrogen_lim)[0]
            self.Q = np.trapz(self.flux[wl_pts], 
                  self.wavelength[wl_pts])/units.h_cgs/(units.c/units.Angstrom)
        
        pass
    def compute_line_emission(self):                        
        if self.line =='hbeta':                  ## Hb=4861 AA       
            self.L_em=self.Q*self.jB/self.alfB     
        if self.line =='halpha':                 ## Ha=6563 AA 
            self.L_em=self.Q*self.a_over_b[self.Z]\
            *self.jB/self.alfB  
            
        
    def alf_B(self):       # Ferland 1980    
        if self.Te <= 2.6*10**4:
            self.alfB=2.90*pow(10,-10)*pow(self.Te,-0.77)         
        else:
            self.alfB=1.31*pow(10,-8)*pow(self.Te,-1.13)    # [cm³/s]    
        
    
    def j_B(self):         # Ferland 1980
        if self.Te <= 2.6*10**4:
            self.jB=2.53*pow(10,-22)*pow(self.Te,-0.833)     #/(4*np.pi)  
        else:
            self.jB=1.12*pow(10,-20)*pow(self.Te,-1.20)     #/(4*np.pi)     # [erg/cm³/s]    
    
    def compute_EW(self):
        if self.line=='hbeta':        
            blue_band =  [4827.875, 4847.875]
            red_band = [4876.625, 4891.625]
            central = [4847.875, 4876.625]
            
        if self.line=='halpha':           
            blue_band =  [6510, 6530]
            red_band = [6600., 6620.]
            central = [6545., 6580.]
            
            
        # Take all points within the blue band: lambda<lambda_blue_2     
        left_lamb_pos = np.where((self.wavelength>=blue_band[0])&(self.wavelength<=blue_band[-1]))[0]           
        mean_left_flux = np.nanmean(
                self.flux[left_lamb_pos]/self.wavelength[left_lamb_pos])
        lamb_left = (blue_band[0]+blue_band[-1])/2
        self.lamb_left = lamb_left
#        self.blue_error =  [np.nanstd(self.flux[left_lamb_pos]),
#                            len(left_lamb_pos)]
        delta_Cb = np.sqrt(np.nanmean(self.flux_err[left_lamb_pos]**2))
        
        # Take all points within the red band: lambda>lambda_red_1
        right_lamb_pos = np.where((self.wavelength>=red_band[0])&(self.wavelength<=red_band[-1]))[0]    
        mean_right_flux = np.nanmean(
                self.flux[right_lamb_pos]/self.wavelength[right_lamb_pos])    
        lamb_right = (red_band[0]+red_band[-1])/2
        self.lamb_right = lamb_right
#        self.red_error =  [np.nanstd(self.flux[right_lamb_pos]), 
#                           len(right_lamb_pos)]
        delta_Cr = np.sqrt(np.nanmean(self.flux_err[right_lamb_pos]**2))
        
        # Fit to a straight line        
#        lamb_line = (lamb_left, lamb_right)
#        A = np.vstack([lamb_line, np.ones_like(lamb_line)]).T        
#        m, c = np.linalg.lstsq(A, [mean_left_flux, mean_right_flux])[0]
        
        delta_lamb = lamb_right-lamb_left
                
        self.mean_left_flux = mean_left_flux
        self.mean_right_flux = mean_right_flux
#        self.pseudocont = lambda lamb: c+m*lamb
        self.pseudocont = lambda lamb: mean_left_flux*((lamb_right-lamb)/delta_lamb)\
        +mean_right_flux*(lamb-lamb_left)/delta_lamb        
            
        self.pseudocont_err = lambda lamb: delta_Cb*\
        ((lamb_right-lamb)/delta_lamb)+\
        delta_Cr*(lamb-lamb_left)/delta_lamb            
            
                
        central_lamb_pts = np.where((self.wavelength>=central[0])&(
                self.wavelength<=central[-1]))[0]
        
        central_lamb = self.wavelength[central_lamb_pts]
        
        central_flux = self.flux[central_lamb_pts]/central_lamb
        central_flux = central_flux+self.L_em/len(central_flux)
        
        central_flux_err = self.flux_err[central_lamb_pts]
        
        central_pseudocont = self.pseudocont(central_lamb)
        self.central_pseudo = central_pseudocont
        self.centra_wl =central_lamb
        central_pseudocont_err = self.pseudocont_err(central_lamb)
        
        central_lamb_int = np.insert(central_lamb, 0,
                                     2*central_lamb[0]-central_lamb[1])                                           
        delta_lamb  = np.ediff1d(central_lamb_int)
        
#        self.lick_index = np.trapz(1-central_flux/central_pseudocont, 
#                                   central_lamb)
        self.EW = np.trapz(1-central_flux/central_pseudocont, central_lamb)
        
        
        self.EW_err = np.nansum(
                (delta_lamb*central_flux_err/central_pseudocont)**2) +\
                np.nansum(
                (delta_lamb*central_pseudocont_err*central_flux/
                                       central_pseudocont**2)**2)
        
        self.EW_err = np.sqrt(self.EW_err)
