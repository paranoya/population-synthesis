#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:02:03 2020

@author: pablo
"""

import numpy as np
import pylab as plt

import basic_properties as SPiCE_galaxy
import chemical_evolution_2 as SPiCE_CEM
import SSP as SPiCE_SSP
import sys
sys.path.append("..")
import units
import os
import pandas as pd

#from DUST_CODE import Dust_model


#plt.switch_backend('AGG') # Default (interactive) backend crashes the server the *second* time (!!!???)

# =============================================================================
# INPUT MODELS
# =============================================================================

#-----------------------------------------------------------------------------
SFH = 'EXP_delayed_tau_BC'
if os.path.isdir(SFH)==False:
    os.mkdir(SFH)
    print('Directory created: \n   ', SFH)

marta_table = np.array([[0.0,  0.036, 0.490, 0.024],
               [0.0001, 0.029, 0.482, 0.019],
               [0.0004, 0.031, 0.486, 0.020],
               [0.004,  0.033, 0.487, 0.020],
               [0.008,  0.034, 0.486, 0.021],
               [0.02,  0.037, 0.483, 0.022],        
               [0.05,  0.041, 0.465, 0.024]])

ejected_met = np.mean(marta_table[:,1])
returned_gas_frac = np.mean(marta_table[:,2])
ox_frac = np.mean(marta_table[:,3])
z_sn = ejected_met/returned_gas_frac
w = 1
M_inf = 1

tau0 =  np.logspace(-1, -0.01, 40)    
tau1 =  np.linspace(1, 5, 40)    
tau2 =  np.logspace(0.7, 1.7, 40)    
tau = np.concatenate((tau0, tau1, tau2))
# Internal parameters
t_today = 13.7
galaxy = SPiCE_galaxy.Point_source(today=t_today, R0=100)


# =============================================================================
# # Output's path
# =============================================================================
SFHfolder = SFH+'/'
SEDfolder= SFHfolder + 'epoch'+str(t_today)+'Gyr_long'         # For further classification
if os.path.isdir(SEDfolder)==False:
    os.mkdir(SEDfolder)
    os.mkdir(SEDfolder+'/'+'Mass&Z')
    print('Directory created: \n   ', SEDfolder)
SEDfolder = SEDfolder+'/'


# today: Time of observation, in Gyr
# R0: Radius of the wind-blown cavity, in pc

# SSPs Initial Mass Function:
#IMF =  ["sal_0.15_100", "fer_0.15_100", "kro_0.15_100", "cha_0.15_100"]:
IMF = ["chab"]


# =============================================================================
# MAIN CODE
# =============================================================================


def compute_SED( galaxy, evolution, SSP ):    

    """ Analytically: L_nu(t_obs) = int { SFR(t_obs -t_ssp) * 
    L_nu_ssp(Z(t_obs -t_ssp),t_ssp) dt_ssp  }"""
    
    t_i = SSP.log_ages_yr - np.ediff1d( SSP.log_ages_yr, to_begin=0 )/2   
    t_i = np.append( t_i, 12 ) # 1000 Gyr
    t_i = galaxy.today - ( units.yr * np.power(10,t_i) )
    t_i[0] = galaxy.today   
    t_i.clip( 0.00001, galaxy.today, out=t_i )    
    M_i = evolution.integral_SFR( t_i )
    M_i = -np.ediff1d( M_i )
    Z_i = evolution.integral_Z_SFR( t_i )
    Z_i = -np.ediff1d( Z_i ) / (M_i+units.kg)    
    Z_i.clip( SSP.metallicities[0], SSP.metallicities[-1], out=Z_i ) # to prevent extrapolation
    SED=np.zeros( SSP.wavelength.size ) 
    for i, mass in enumerate(M_i):
        #print(t_i[i]/units.Gyr, SSP.log_ages_yr[i],'\t', m/units.Msun, Z_i[i])
        if mass>0:
            index_Z_hi = SSP.metallicities.searchsorted( Z_i[i] ).clip( 1, len(SSP.metallicities)-1 )
            weight_Z_hi = np.log( Z_i[i]/SSP.metallicities[index_Z_hi-1] ) / np.log( SSP.metallicities[index_Z_hi]/SSP.metallicities[index_Z_hi-1] ) # log interpolation in Z
            SED = SED + mass*( SSP.SED[index_Z_hi][i]*weight_Z_hi + SSP.SED[index_Z_hi-1][i]*(1-weight_Z_hi) ) # Sed(t)  is integrated over the SSP ages   
                
    return SED

  
all_stellar_age = []
all_z_star = []
all_z_gas = []
all_ssfr= []
all_youngmass = []
all_star_to_gas = []
all_oh = []


# POPULATION SYNTHESIS:
    
for IMF_i in IMF:
  SSP = SPiCE_SSP.BC03_Padova94(mode='hr', IMF=IMF_i)
        
  for i, tau_i in enumerate(tau): 
  
      CEM = SPiCE_CEM.Exponential_delayed_tau(M_inf=M_inf, tau=tau_i,
                                        R=0.5, z_sn=z_sn,  w=w)
          
      filename = IMF_i + '_tau_'+format(tau_i, '08.4f')
      
      print(filename)
    
      CEM.set_current_state( galaxy )
      
      t = np.linspace(1, galaxy.today,num=101)
      
      Mass = CEM.integral_SFR(t)
      Mass_gas = CEM.M_gas(t)
      Mass_metals = CEM.M_metals(t)
      
      stellar_mass_500myr = CEM.integral_SFR(galaxy.today-500*units.Myr)
      young_mass_frac = (Mass[-1]-stellar_mass_500myr)/Mass[-1]
      print(young_mass_frac)
      ssfr = CEM.sSFR(t)*units.yr
      
      met_Z = CEM.integral_Z_SFR(t)      
      met_Z = met_Z/Mass
      met_Z_gas = CEM.Z_gas(t)      
      oxygen_abundance = CEM.oxygen_abundance(met_Z_gas[-1])
      
      int_t_sfr = CEM.integral_t_SFR(t)
      stellar_age = galaxy.today - int_t_sfr/Mass
      
      all_stellar_age.append(stellar_age[-1]/units.Gyr)
      all_z_star.append(met_Z[-1])
      all_z_gas.append(met_Z_gas[-1])
      all_oh.append(oxygen_abundance)
      all_ssfr.append(ssfr[-1])
      all_youngmass.append(young_mass_frac)
      all_star_to_gas.append(Mass[-1]/Mass_gas[-1])

      Mass /= units.Msun
      Mass_gas /= units.Msun            
      Mass_metals /= units.Msun            
      
      t /= units.Gyr
      if Mass.max() > 0.:
              
          fig = plt.figure()
          ax = plt.subplot(111)
          ax.set_xlabel(r't [Gyr]')
          ax.set_ylabel(r'$M/M_0$')
          ax.set_xscale('log')
          ax.set_yscale('log')
          ax.set_ylim( [1e-6, 3] )
          ax.set_xlim( [1e-1, t[-1]] )
          ax.plot( t, Mass, 'k-*', figure=fig )
          ax.plot( t, Mass_metals, 'k--', figure=fig )
          ax.plot( t, Mass_gas, 'k-', figure=fig )
          ax1=ax.twinx()
          ax1.plot( t, met_Z, 'b-*', figure=fig )
          ax1.plot( t, met_Z_gas, 'c-', figure=fig )
          ax1.set_ylabel(r'$Z$')
          ax1.spines['right'].set_color('blue')
          ax1.tick_params(axis='y', colors='blue')
          ax1.yaxis.label.set_color('blue')
          fig.tight_layout()
          fig.savefig(SEDfolder+'Mass&Z/'+'SED_'+filename+'.png')
          plt.close(fig)
      
         
      SED = compute_SED( galaxy, CEM, SSP  )
      
      nu_Lnu_erg_s_stel = SSP.wavelength*SED / (units.erg/units.second)/Mass[-1]
      nu_Lnu_erg_s_neb = SSP.wavelength*SED / (units.erg/units.second)/Mass[-1]
      nu_Lnu_erg_s_Tot = SSP.wavelength*SED / (units.erg/units.second)/Mass[-1]
      
      l_A = SSP.wavelength/units.Angstrom
      nu = units.c/SSP.wavelength
    
      L_lambda0=SED/(3.828e33*units.erg/units.second/units.Angstrom)          # stel
      L_lambda1=SED/(3.828e33*units.erg/units.second/units.Angstrom)          # neb
      L_lambda2=SED/(3.828e33*units.erg/units.second/units.Angstrom)          # Tot
      
      sed = zip( l_A, nu, nu_Lnu_erg_s_stel, nu_Lnu_erg_s_neb , nu_Lnu_erg_s_Tot, L_lambda0, L_lambda1, L_lambda2)
      
      
    
      if nu_Lnu_erg_s_stel.max() > 0.:
    	       fig = plt.figure()
    	       ax = plt.subplot(111)
    	       ax.set_xlabel(r'$\lambda$ [$\AA$]')
    	       ax.set_ylabel(r'$\nu$ L$_\nu$ [erg/s/$M_\odot$]')
    	       ax.set_xlim( [100, 1e5] )
    	       ax.set_ylim( [1e-6*nu_Lnu_erg_s_stel.max(), 2*nu_Lnu_erg_s_stel.max()] )
    	       ax.set_title(filename)
    	       plt.loglog(l_A, nu_Lnu_erg_s_stel, figure=fig)
      ax.annotate('Z={:.5}'.format(met_Z[-1]), xy=(.1,.1), xycoords='axes fraction')                              
      plt.savefig(SEDfolder+'/'+filename+'.png')
      plt.close(fig)                 
    	
      with open( SEDfolder +'SED_'+filename+'.txt', 'w' ) as f:
    	          f.write('# wavelength [AA] ---- Frec [HZ] ---- STELLAR nu*f_nu [erg/s] ---- NEBULAR nu*f_nu [erg/s] ---- Total nu*f_nu [erg/s] ---- f_lambda_stellar[erg/s/AA] ---- f_nu_neb ----f_nu_Tot \n ')
    	          for ll, nn, ss, dd, tt, zz, kk, jj in sed:
    	              f.write('{:.5} {:.5} {:.5} {:.5} {:.5} {:.5} {:.5} {:.5}\n'.format(ll, nn, ss, dd, tt, zz, kk, jj))


# =============================================================================
# Global parameters    
# =============================================================================
globparams_folder = SEDfolder+'global_params'
if os.path.isdir(globparams_folder)==False:
    os.mkdir(globparams_folder)
    print('Directory created: \n   ', globparams_folder)
globparams_folder += '/'

data = {'tau':tau,
        'stellar_age':all_stellar_age,
        'Z_stars':all_z_star,
        'Z_gas':all_z_gas,
        'oh':all_oh,
        'lg_sSFR':np.log10(all_ssfr),
        'lg_young_mass_frac':np.log10(all_youngmass),
        'lg_s_to_g':np.log10(all_star_to_gas)
        }                      
df = pd.DataFrame(data=data)                      
df.to_csv(globparams_folder+'params_t_{:.2}.csv'.format(galaxy.today/units.Gyr))

print('\n SED computation finished \n')
