#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../numerics-statistics")
import units
from integral_interpolator import simpson_integral_1d as num_int
from integral_interpolator import num_int


#-------------------------------------------------------------------------------
class Chemical_evolution_model:        
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
     pass
#    self.M_gas = kwargs['M_gas']*units.Msun
#    self.Z = kwargs['Z']
    
  def get_Z(self, time):
    return self.Z(time)
  
  def set_current_state(self, galaxy):
    galaxy.M_gas = self.M_gas( galaxy.today )
    galaxy.Z = self.Z( galaxy.today )
    galaxy.M_stars = self.M_stars( galaxy.today ) #TODO: account for stellar death
    
#-------------------------------------------------------------------------------
class Single_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------
 
  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']*units.Msun          
    self.t = kwargs['t']*units.Gyr                      # Born time 
    Chemical_evolution_model.__init__(self, **kwargs)
  
  def integral_SFR(self, time):
    M_t = np.array(time)
    for i, t in np.ndenumerate(time):
      if t<=self.t:
 	      M_t[i] = 0
      else:
	      M_t[i] = self.M_stars
    return M_t

  def integral_Z_SFR(self, time):
    Z_t = np.array(time)
    for i, t in np.ndenumerate(time):
      if t<=self.t:
	      Z_t[i] = 0
      else:
	      Z_t[i] = self.Z * self.M_stars
    return Z_t
  
#-------------------------------------------------------------------------------
class Exponential_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)
      
  def integral_SFR(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau) )

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Exponential_SFR_delayed(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr       
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)
      
  def integral_SFR(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau)*(self.tau+time)/self.tau)

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Exponential_quenched_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr
    self.Z = kwargs['Z']
    self.t_q = kwargs['t_quench']*units.Gyr
    Chemical_evolution_model.__init__(self, **kwargs)
    

  def integral_SFR(self, time):
     if type(time) is float:
        if time<self.t_q:
              M_stars=self.M_inf * ( 1 - np.exp(-time/self.tau) )
                
        else:
            M_stars=self.M_inf * ( 1 - np.exp(-self.t_q/self.tau) )
         
     else:    

         M_stars = np.zeros_like(time)
         before_quench = np.where(time<self.t_q)[0]
         after_quench = np.where(time>=self.t_q)[0]
         M_stars[before_quench] = self.M_inf * ( 1 - np.exp(-time[before_quench]/self.tau))
         M_stars[after_quench] = self.M_inf * ( 1 - np.exp(-self.t_q/self.tau) )             
             
     return M_stars

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Powlaw(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  """
  Power-law SFH model with metallicity evolution (IRA) due to gas infall and 
  galactic winds (\propto \omega*SFR).
  
  input params:
      - M_inf: Mass-Normalizing constant.
      - t_0: Today's time.
      - beta: pow-law slope.
      - R: recycled gas fraction from stars.
      - w: galactic wind intensity.
      - Z_sn: Supernova ejecta metallicity.
      
      
  """
  def __init__(self, **kwargs):
    
    self.M_inf = kwargs['M_inf']*units.Msun    
    self.b = kwargs['beta']
    self.t_0 = kwargs['t_0']*units.Gyr        
    self.R = kwargs['R']    
    self.w = kwargs['w']
    self.z_sn = kwargs['z_sn']
            
    self.Z = self.Z_gas
    self.Lambda = (1+self.w)/(1-self.R)
    
    Chemical_evolution_model.__init__(self, **kwargs)

  def sfr(self, time):                    
        sfr = self.b * self.M_inf/self.t_0 *(time/self.t_0)**(self.b -1)
        return sfr      
    
  def M_stars(self, time):
        return self.M_inf * (time/self.t_0)**self.b*(1-self.R) 
    
  def sSFR(self, time):        
        ssfr = self.sfr(time)/self.M_stars(time)         # s^-1
        return ssfr
    
  def M_gas(self, time):                      
        gas = self.tau_gas(time)*(1-self.R)*self.sfr(time)        
        return gas  
    
  def tau_gas(self, time):
#        tau_gas = -1.02*np.log10(self.sSFR(time)) + 0.31
#        tau_gas = 10**tau_gas 
        tau_gas = 10*units.Gyr
        return tau_gas
    
  def M_metals(self, time):                  
      if type(time) is not np.ndarray:            
            time = np.linspace(0, time)            
            metals = self.z_sn*self.R*\
            np.exp(-time*self.Lambda/self.tau_gas(time))*\
            num_int(self.sfr(time)*np.exp(time*self.Lambda/self.tau_gas(time)),
            time)                                    
            metals = metals[-1]            
      else:
            metals = self.z_sn*self.R*\
            np.exp(-time*self.Lambda/self.tau_gas(time))*\
            num_int(self.sfr(time)*np.exp(time*self.Lambda/self.tau_gas(time)),
            time,
            interp=True)                                
      return metals

  def Z_gas(self, time):                    
        Z = self.M_metals(time)/self.M_gas(time)        
        return Z
    
  def Z_stars(self, time):
        z_stars = self.integral_Z_SFR(time)/self.integral_SFR
        return z_stars  
    
  def integral_SFR(self, time):
        return self.M_inf*(time/self.t_0)**self.b
  
  def integral_Z_SFR(self, time):
        int_sfr_z = num_int(self.Z_gas(time)*self.sfr(time) ,time)
        return int_sfr_z

  def integral_t_SFR(self,time):
        int_t_SFR = num_int(time*self.sfr(time) ,time)
        return int_t_SFR


#-------------------------------------------------------------------------------
class Exponential_alpha(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  """
  Exponential-decaying SFH model with metallicity evolution (IRA) due to gas infall 
  (\propto exp(-t/tau)) and galactic winds (\propto \omega*SFR).
  
  input params:
      - phi_0: SFR-Normalizing constant.
      - R: recycled gas fraction from stars.
      - w: galactic wind intensity.
      - Z_sn: Supernova ejecta metallicity.
      - Tau: characteristic timescale for gas consumption.
      
  """
  def __init__(self, **kwargs):
        
    self.alpha = kwargs['alpha']/units.Gyr
    self.phi_0 = kwargs['phi_0']*units.Msun/units.Gyr        
    self.R = kwargs['R']    
    self.w = kwargs['w']
    self.z_sn = kwargs['z_sn']
            
    self.Z = self.Z_gas
    self.Lambda = (1+self.w)/(1-self.R)
    
    Chemical_evolution_model.__init__(self, **kwargs)

  def sfr(self, time):        
        sfr = self.phi_0*np.exp(self.alpha*time)
        return sfr      
    
  def M_stars(self, time):
        return self.phi_0/self.alpha  * (np.exp(self.alpha*time)-1)
    
  def sSFR(self, time):        
        ssfr = self.sfr(time)/self.M_stars(time)         # s^-1
        return ssfr
    
  def M_gas(self, time):  
        gas = self.tau_gas(time)*(1-self.R)*self.sfr(time)        
        return gas  
    
  def tau_gas(self, time):
#        tau_gas = -1.02*np.log10(self.sSFR(time)*units.yr) + 0.31
#        tau_gas = 10**tau_gas *units.yr
        tau_gas = 10*units.Gyr
        return tau_gas
    
  def M_metals(self, time):            
        metals = self.z_sn*self.R*self.phi_0/(1-self.R)\
        /(self.alpha+self.Lambda/self.tau_gas(time)) * (
                np.exp(self.alpha*time) -np.exp(-time*self.Lambda/self.tau_gas(time)))        
        return metals

  def Z_gas(self, time):                    
        cte = self.z_sn*self.R/(1-self.R)/(self.alpha*self.tau_gas(time)+self.Lambda)        
        Z = cte*(1- np.exp(-time*(self.alpha +self.Lambda/self.tau_gas(time))))        
        return Z
      
  def Z_stars(self, time):
        z_stars = self.z_sn*self.R/(1-self.R)/self.Lambda
        z_stars *= 1/(self.alpha*self.tau_gas(time)+self.Lambda)
        z_stars *= (self.Lambda*(np.exp(self.alpha*time)-1)- \
        self.alpha*self.tau_gas(time)*(1-np.exp(-time*self.Lambda/self.tau_gas(time))))
        z_stars /= (np.exp(self.alpha*time)-1)
        return z_stars  
    
  def integral_SFR(self, time):
    return self.M_stars(time)
  
  def integral_SFR_numerical(self, time):
    int_sfr = num_int(self.sfr(time), time)
    return int_sfr

  def integral_Z_SFR(self, time):
    cte =self.z_sn*self.R*self.phi_0/(1-self.R)/(self.alpha*self.tau_gas(time)+self.Lambda)
    int_sfr_z = cte * (self.Lambda*(np.exp(self.alpha*time)-1) -\
           self.alpha*self.tau_gas(time)*(1-np.exp(-time*self.Lambda/self.tau_gas(time))))
    int_sfr_z /= self.alpha*self.Lambda
    return int_sfr_z

  def integral_Z_SFR_numerical(self, time):
    int_sfr_z = num_int(self.Z_gas(time)*self.sfr(time), time)
    return int_sfr_z

  def integral_t_SFR(self,time):
     int_t_SFR = self.phi_0/self.alpha**2*(1+np.exp(self.alpha*time)*(self.alpha*time-1))
     return int_t_SFR


#-------------------------------------------------------------------------------
class Exponential_delayed_alpha(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  """
  Exponential-decaying SFH model with metallicity evolution (IRA) due to gas infall 
  (\propto exp(-t/tau)) and galactic winds (\propto \omega*SFR).
  
  input params:
      - phi_0: SFR-Normalizing constant.
      - R: recycled gas fraction from stars.
      - w: galactic wind intensity.
      - Z_sn: Supernova ejecta metallicity.
      - Tau: characteristic timescale for gas consumption.
      
  """
  def __init__(self, **kwargs):
        
    self.alpha = kwargs['alpha']/units.Gyr
    self.phi_0 = kwargs['phi_0']*units.Msun/units.Gyr**2        
    self.R = kwargs['R']    
    self.w = kwargs['w']
    self.z_sn = kwargs['z_sn']
            
    self.Z = self.Z_gas
    
    
#    self.tau_gas = 10*units.Gyr
    self.Lambda =lambda time: (1+self.w)/self.tau_gas(time)
    
    Chemical_evolution_model.__init__(self, **kwargs)
    
  def sfr(self, time):        
        sfr = self.phi_0*time*np.exp(self.alpha*time)
        return sfr      
    
  def M_stars(self, time):
        m_stars= self.integral_SFR(time)*(1-self.R) + units.kg
        return m_stars
    
  def sSFR(self, time):        
        ssfr = self.sfr(time)/self.M_stars(time)         # s^-1
        return ssfr
    
  def M_gas(self, time):  
        gas = self.tau_gas(time)*self.sfr(time) + units.kg       
        return gas  
    
  def tau_gas(self, time):
        tau_gas = -1.02*np.log10(self.sSFR(time)*units.yr+1/units.yr) + 0.31
        tau_gas = 10**tau_gas *units.yr         
        return tau_gas
    
  def M_metals(self, time):            
        metals = self.z_sn*self.R*self.phi_0/(self.alpha+self.Lambda(time))**2 *\
        (np.exp(time*self.alpha)*(time*(self.alpha+self.Lambda(time))-1)+ np.exp(-time*self.Lambda(time)))        
        return metals

  def Z_gas(self, time):                    
        Z = self.M_metals(time)/self.M_gas(time)        
        return Z
    
  def Z_stars(self, time):
        z_stars = self.integral_Z_SFR(time)/self.integral_SFR(time) 
        return z_stars  
    
  def integral_SFR(self, time):
    return self.phi_0/self.alpha**2 *(1+np.exp(self.alpha*time)*(self.alpha*time-1))
  
  def integral_Z_SFR(self, time):
#    cte =self.z_sn*self.R*self.phi_0/(1-self.R)/(self.alpha*self.tau_gas(time)+self.Lambda)
#    int_sfr_z = cte * (self.Lambda*(np.exp(self.alpha*time)-1) -\
#           self.alpha*self.tau_gas(time)*(1-np.exp(-time*self.Lambda/self.tau_gas(time))))
#    int_sfr_z /= self.alpha*self.Lambda
    int_sfr_z = num_int(self.Z_gas(time)*self.sfr(time), time)      
    return int_sfr_z

  def integral_t_SFR(self,time):
     int_t_SFR = num_int(time*self.sfr(time), time)      
     return int_t_SFR
 
#-------------------------------------------------------------------------------
class Exponential_delayed_tau(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  """
  Exponential-decaying SFH model with metallicity evolution (IRA) due to gas infall 
  (\propto exp(-t/tau)) and galactic winds (\propto \omega*SFR).
  
  input params:
      - M_inf: Mass-Normalizing constant.
      - R: recycled gas fraction from stars.
      - w: galactic wind intensity.
      - Z_sn: Supernova ejecta metallicity.
      - Tau: characteristic timescale for gas consumption.
      
  """
  def __init__(self, **kwargs):
        
    self.tau = kwargs['tau']*units.Gyr
    self.M_inf = kwargs['M_inf']*units.Msun        
    self.R = kwargs['R']    
    self.w = kwargs['w']
    self.z_sn = kwargs['z_sn']
            
    self.Z = self.Z_gas
        
    self.Lambda = self.R/(1+self.w-self.R)/self.tau

    
    Chemical_evolution_model.__init__(self, **kwargs)
    
  
    

    
  def sSFR(self, time):        
        ssfr = self.sfr(time)/self.M_stars(time)         # s^-1
        return ssfr
    
  def M_gas(self, time):  
        gas = self.M_inf * time/self.tau * np.exp(-time/self.tau)   
        return gas  
  
  def sfr(self, time):        
        sfr = self.M_gas(time)/(self.tau*(1-self.R+self.w))
        return sfr        

  def M_stars(self, time):
        m_stars= self.M_inf*(1-self.R)/(1-self.R +self.w) * (1-np.exp(-time/self.tau)*(1+time/self.tau))
        return m_stars    
    
  def M_metals(self, time):            
        metals = self.z_sn*self.M_inf*self.R/(1-self.R+self.w)/self.tau**2/self.Lambda**2 \
        * (np.exp(-time/self.tau)*(time*self.Lambda-1) \
           + np.exp(-time/self.tau *(1+self.w)/(1-self.R+self.w))
           )
        return metals

  def Z_gas(self, time):                    
        Z = self.M_metals(time)/self.M_gas(time)        
#        Z = self.z_sn*self.R/((1+self.w-self.R)*(1-self.Lambda)**2)*self.tau/time \
#        *( 1-time*(1-self.Lambda)/self.tau + np.exp(-time*(self.Lambda-1)/self.tau))                  
        
        return Z
    
  def Z_stars(self, time):
        z_stars = self.integral_Z_SFR(time)/self.integral_SFR(time) 
        return z_stars  
 
  def oxygen_abundance(self, Z):        
        oxygen_over_metals = 0.51
        hydrogen_abundance = 0.739
        oxygen_Z = 16
        model_oh = 12 + np.log10(Z/oxygen_Z/hydrogen_abundance) + np.log10(oxygen_over_metals)
        return model_oh  
    
  def integral_SFR(self, time):
    return self.M_inf/(1-self.R +self.w)*(1-np.exp(-time/self.tau)*(time/self.tau+1))
  
  def integral_Z_SFR(self, time):
#    cte =self.z_sn*self.R*self.phi_0/(1-self.R)/(self.alpha*self.tau_gas(time)+self.Lambda)
#    int_sfr_z = cte * (self.Lambda*(np.exp(self.alpha*time)-1) -\
#           self.alpha*self.tau_gas(time)*(1-np.exp(-time*self.Lambda/self.tau_gas(time))))
#    int_sfr_z /= self.alpha*self.Lambda
    int_sfr_z = num_int(self.Z_gas(time)*self.sfr(time), time)      
    return int_sfr_z

  def integral_t_SFR(self,time):
     int_t_SFR = num_int(time*self.sfr(time), time)      
     return int_t_SFR    

#-------------------------------------------------------------------------------
class Generic_model(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  """
  This model computes numerically the chemical evolution of a models with 
  a given SFH under IRA.
  
  input params:
      - SFH(t) (method)
      - R: recycled gas fraction from stars.
      - w: galactic wind intensity.
      - Z_sn: Supernova ejecta metallicity.
      - Tau: characteristic timescale for gas consumption (10 Gyr by default).
      
  """
  def __init__(self, **kwargs):
    
    self.sfr = kwargs['sfr']        
    self.R = kwargs['R']    
    self.w = kwargs['w']
    self.z_sn = kwargs['z_sn']
            
    self.Z = self.Z_gas
    self.Lambda = (1+self.w)/(1-self.R)
    
    
    Chemical_evolution_model.__init__(self, **kwargs)
  
  
  def integral_SFR(self, time):
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)
          int_sfr = num_int(self.sfr(time), time) +units.kg
          return int_sfr[-1]      
      else:    
          int_sfr = num_int(self.sfr(time), time) +units.kg
          return int_sfr    
    
  def M_stars(self, time):
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)
          m_stars = (1-self.R)*self.integral_SFR(time)
          return m_stars[-1]
      else:    
          return (1-self.R)*self.integral_SFR(time)
    
  def sSFR(self, time):        
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)
          ssfr = self.sfr(time)/self.M_stars(time)       # s^-1
          return ssfr[-1]
      else:    
          ssfr = self.sfr(time)/self.M_stars(time)         # s^-1
          return ssfr
    
  def M_gas(self, time):  
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)
          gas = self.tau_gas(time)*self.sfr(time) +units.kg        
          return gas[-1]
      else:    
        gas = self.tau_gas(time)*self.sfr(time) +units.kg        
        return gas  
    
  def tau_gas(self, time):
#        tau_gas = -1.02*np.log10(self.sSFR(time)) + 0.31
#        tau_gas = 10**tau_gas 
        tau_gas = 10*units.Gyr
        return tau_gas
    
  def M_metals(self, time):            
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)
          metals = self.z_sn*self.R*np.exp(-time*self.Lambda/self.tau_gas(time))*\
          num_int(self.sfr(time)*np.exp(time*self.Lambda/self.tau_gas(time)), time)
          return metals[-1]
      else:    
        metals = self.z_sn*self.R*np.exp(-time*self.Lambda/self.tau_gas(time))*\
        num_int(self.sfr(time)*np.exp(time*self.Lambda/self.tau_gas(time)), time)
        return metals

  def Z_gas(self, time):              
      if type(time) is not np.ndarray:
          time = np.linspace(0, time)      
          Z = self.M_metals(time)/self.M_gas(time)
          return Z[-1]
      else:    
          Z = self.M_metals(time)/self.M_gas(time)
          return Z

  def integral_Z_SFR(self, time):
        int_sfr_z = num_int(self.Z_gas(time)*self.sfr(time), time)
        return int_sfr_z

      
  def Z_stars(self, time):
        z_stars = self.integral_Z_SFR(time)/self.integral_SFR(time)        
        return z_stars  
  

  def integral_t_SFR(self,time):
     int_t_SFR = num_int(self.sfr(time)*time, time)
     return int_t_SFR


#-------------------------------------------------------------------------------
#class MD05(Chemical_evolution_model):
##-------------------------------------------------------------------------------
#  path = os.path.join( os.path.dirname(__file__), 'data/MD05' )
#  
#  def __init__(self, **kwargs):
#    filename = kwargs['V_rot']+kwargs['eff']
#    self.R = kwargs['R']
#    while self.R > 0 and not os.path.exists( os.path.join(MD05.path,'{}radius_{:02.0f}'.format(filename,self.R)) ):
#      self.R -=1
#    filename = os.path.join(MD05.path,'{}radius_{:02.0f}'.format(filename,self.R))
#    print("> Reading MD05 file: '"+ filename +"'" , t, dm, Z = np.loadtxt( filename, dtype=np.float, unpack=True)
#    self.t_table =np.append([0],(t+0.5)*units.Gyr )
#    self.Z_table = np.append( [0], Z )
#    dm = np.append( [0], dm*1e9*units.Msun )
#    self.integral_SFR_table = np.cumsum(dm)
#    self.integral_Z_SFR_table = np.cumsum( self.Z_table*dm )
#    
#  def set_current_state(self, galaxy):
#    if self.R > 0.5:
#      area = 2*np.pi*self.R*units.kpc**2
#    else:
#      area = np.pi*((self.R+0.5)*units.kpc)**2
#    fraction_of_area_included = min(galaxy.Area_included/area, 1.)
#    
#    galaxy.M_stars = fraction_of_area_included * self.integral_SFR( galaxy.today ) #TODO: account for stellar death
#    Z = np.interp( galaxy.today, self.t_table, self.Z_table )
#    galaxy.M_gas = galaxy.M_stars*max(.03/Z-1,1e-6) # TODO: read from files
#    galaxy.Z = Z
#    
#  def integral_SFR(self, time):
#    return np.interp( time, self.t_table, self.integral_SFR_table )
#
#  def integral_Z_SFR(self, time):
#    return np.interp( time, self.t_table, self.integral_Z_SFR_table )

#-------------------------------------------------------------------------------
class ASCII_file(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, file,
	       time_column = 0,
	       Z_column    = 1,
	       SFR_column  = 2,
	       time_units  = units.Gyr,
	       SFR_units   = units.Msun/units.yr ):
    print("> Reading SFR file: '"+ file +"'")
    t, Z, SFR = np.loadtxt( file, dtype=np.float, usecols=(time_column,Z_column,SFR_column), unpack=True)
    self.t_table   = np.append( [0], t*time_units )
    self.Z_table   = np.append( [0], Z )

    dt = np.ediff1d( self.t_table, to_begin=0 )
    dm = np.append( [0], SFR*SFR_units )*dt
    self.integral_SFR_table = np.cumsum( dm )
    self.integral_Z_SFR_table = np.cumsum( self.Z_table*dm )
    
  def set_current_state(self, galaxy):
    galaxy.M_stars = self.integral_SFR( galaxy.today ) #TODO: account for stellar death
    Z = np.interp( galaxy.today, self.t_table, self.Z_table )
    galaxy.M_gas = galaxy.M_stars*max(.03/(Z+1e-6)-1,1e-6) # TODO: read from files
    galaxy.Z = Z
    
  def integral_SFR(self, time):
    return np.interp( time, self.t_table, self.integral_SFR_table )

  def integral_Z_SFR(self, time):
    return np.interp( time, self.t_table, self.integral_Z_SFR_table )

  #def plot(self):
    #plt.semilogy( self.t_table/units.Gyr, self.SFR_table/(units.Msun/units.yr) )
    #plt.semilogy( self.t_table/units.Gyr, self.integral_SFR_table/units.Msun )
    #plt.semilogy( self.t_table/units.Gyr, self.Z_table )
    #plt.show()

#-------------------------------------------------------------------------------


# Mr. Krtxo...