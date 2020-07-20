#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:21:05 2019

@author: pablo
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck13, z_at_value
from astropy import units 



def V_max(u, g, r, survey, ugr_lims, z_lims):
    
    cosmo = FlatLambdaCDM(H0=70 * units.km / units.s / units.Mpc, 
                      Tcmb0=2.725 * units.K, Om0=0.3)
    u_limit = ugr_lims[0]
    g_limit = ugr_lims[1]
    r_limit = ugr_lims[2]
    z_min = z_lims[0]
    z_max = z_lims[1]

    D_z_min = cosmo.luminosity_distance(z_min).value #Mpc
    D_z_max = cosmo.luminosity_distance(z_max).value #Mpc
#    H0 = 100  # km/s/Mpc
#    D_z_min = z_min * 3e5/H0
#    D_z_max = z_max * 3e5/H0
    if survey=='SDSS':
        survey_solid_angle = 1317  # deg**2
    elif survey=='GAMA':
        survey_solid_angle = 180  # deg**2
    
    survey_solid_angle *= (np.pi/180)**2  # steradian
        
    D_u = 10**(0.2 * (u_limit-u) - 5)  # Mpc
    D_g = 10**(0.2 * (g_limit-g) - 5)  # Mpc
    D_r = 10**(0.2 * (r_limit-r) - 5)  # Mpc

    D_max = np.min([D_u, D_g, D_r], axis=(0)).clip(D_z_min, D_z_max)
    return survey_solid_angle * (D_max**3 - D_z_min**3) / 3


# =============================================================================
class Ansatz():

    def parameters_given_M(M_star):
        M_sym = 10**10
        tau_sym = 2.16
        alpha_sym = 3.6

        eta_tau = 0.09
        eta1 = -0.3
        eta2 = 1

        tau_0 = tau_sym*(M_star/M_sym)**eta_tau
        alpha1 = -1 + (alpha_sym+1)*(M_star/M_sym)**(eta1)
        alpha2 = 1 + (alpha_sym-1)*(M_star/M_sym)**(eta2)
        beta = 4

        return tau_0, alpha1, alpha2, beta

    def dp_dtau_given_M(M_star, tau):
        """
        Probability density of the characteristic time
        scale tau (scalar or array) for a given stellar mass (must be scalar)
        """
        tau_0, alpha1, alpha2, beta = Ansatz.parameters_given_M(M_star)

        x = tau/tau_0
        dp_dtau = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)

        # Normalize dp_dtau by integrating (do the integral in log tau)
        log_tau_interp = np.linspace(-2, 4, 1000)
        x = 10**log_tau_interp/tau_0
        dp_dtau_interp = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)
        integral_dpdtau_dtau = np.log(10)*tau_0 * np.trapz(dp_dtau_interp*x,
                                                           log_tau_interp)
        return dp_dtau / integral_dpdtau_dtau

    def dp_dtau_grid(M_star, tau):
        tau_0, alpha1, alpha2, beta = Ansatz.parameters_given_M(M_star)

        x = tau[:, np.newaxis]/tau_0[np.newaxis, :]
        dp_dtau = pow(x, alpha1[np.newaxis, :]) / pow(
                1+x**beta, (alpha1[np.newaxis, :]+alpha2[np.newaxis, :])/beta)

        # Normalize dp_dtau by integrating (do the integral in log tau)
        log_tau_interp = np.linspace(-2, 4, 1000)
        x = 10**log_tau_interp[:, np.newaxis]/tau_0[np.newaxis, :]
        dp_dtau_interp = pow(x, alpha1[np.newaxis, :]) / pow(
                1+x**beta, (alpha1[np.newaxis, :]+alpha2[np.newaxis, :])/beta)
        integral_dpdtau_dtau = np.log(10)*tau_0 * np.trapz(
                dp_dtau_interp*x, log_tau_interp, axis=0)
        return dp_dtau / integral_dpdtau_dtau[np.newaxis, :]

    def dn_dM(M, M_schechter=10**(11), alpha=-1.55, phi=0.01):
        """
        Multiplicity function (number density of galaxies
        per unit comoving volume per unit mass), described
        as a Schechter function.
        """
        return phi*np.exp(-(M/M_schechter))/M_schechter * (M/M_schechter)**alpha

    def dn_dMdtau_grid(M, tau):
        """
        Returns number density of galaxies as a function of M and tau
        """
# Alternative way of creating dp_dtau grid:
#        N_tau = len(tau)
#        N_M = len(M)
#        dp_dtau_grid = Ansatz.dp_dtau_unnorm(M.repeat(N_tau), np.tile(tau, N_M))
#        dp_dtau_grid.shape = (N_M, N_tau)
#
#        log_tau = np.log10(tau)
#        d_log_tau = np.concatenate((
#                log_tau[1]-log_tau[0],
#                (log_tau[2:]-log_tau[:-2]) / 2,
#                log_tau[-1]-log_tau[-2]
#                ))
#        norm = np.log(10) * np.sum(
#                dp_dtau_grid * d_log_tau[np.newaxis, :]*tau[np.newaxis, :],
#                axis=1)
        dn_dM = Ansatz.dn_dM(M)
        return dn_dM[np.newaxis, :] * Ansatz.dp_dtau_grid(M, tau)


# =============================================================================
class Model_grid(object):

    def __init__(self, **kwargs):
#        self.input_metallicities = [0.0004, 0.004, 0.008, 0.02, 0.05]
        """
        - Default model input metallicities Z=0.004, 0.008, 0.02, 0.05
            Those that the method load_()mod will search in some directory.
            
        - Defaul stellar mass range: log(M)= 8 , 12 dex.
        
        """
        self.input_metallicities = [0.004, 0.008, 0.02, 0.05]
        self.input_log_Z = np.log10(self.input_metallicities)
        print('Metallicities :', self.input_metallicities)
        self.log_M_star = np.linspace(8, 12, 200)
        self.M_star = 10**self.log_M_star

        
#        try:
        models_path = kwargs['photomod_path']
        self.load_photomod(models_path, n_metallicities_interpolation=25)
#        except:
#            print('No photometry models loaded')
#        try: 
        models_path = kwargs['specmod_path']
        self.load_specmod(models_path, n_metallicities_interpolation=25)
#        except:
#            print('No spectroscopic models loaded')
            
#        self.compute_V_max()
#        self.compute_N_galaxies()

    def load_photomod(self, path, n_metallicities_interpolation=0):

        tau_model = []
        u_model = []
        g_model = []
        r_model = []
        i_model = []
        z_model = []

        for Z_i in self.input_metallicities:  # TODO: Include IMF
            tau, u, g, r, i, z = np.loadtxt(
                    path +
                    'photometry_Z_{:.4f}.txt'.format(Z_i),
                    skiprows=0, usecols=(0, 2, 3, 4, 5, 6), unpack=True) 
                    # 1 col corresponds to the extinction value
            tau_model.append(tau.reshape(300,30))  # axis 1 correspond to the 
                                                   # dust ext. array
            u_model.append(u.reshape(300,30))
            g_model.append(g.reshape(300,30))
            r_model.append(r.reshape(300,30))
            i_model.append(i.reshape(300,30))
            z_model.append(z.reshape(300,30))
            

        
        self.tau = tau_model[-1][:, -1]
        self.log_tau = np.log10(self.tau)
        print('tau_min = {:.2f}, tau_max = {:.2f}'.format(
                min(self.tau), max(self.tau)))
        self.metallicities = self.input_metallicities
        self.log_Z = np.log10(self.metallicities)
        # Define range of metallicities
        if n_metallicities_interpolation > 0:
#            new_metallicities = np.linspace(self.input_log_Z[0], self.input_log_Z[-1],
#                                            n_metallicities_interpolation)
            new_metallicities = np.log10(np.linspace(
                    self.input_metallicities[0],
                    self.input_metallicities[-1],
                    n_metallicities_interpolation)
                                        )
            
            u_model = interp1d(self.input_log_Z, u_model, axis=0)(new_metallicities)
            g_model = interp1d(self.input_log_Z, g_model, axis=0)(new_metallicities)
            r_model = interp1d(self.input_log_Z, r_model, axis=0)(new_metallicities)
            i_model = interp1d(self.input_log_Z, i_model, axis=0)(new_metallicities)
            z_model = interp1d(self.input_log_Z, z_model, axis=0)(new_metallicities)

            self.metallicities = 10**new_metallicities
            self.log_Z = new_metallicities
            print('New range of metallicities: \n', new_metallicities)

        # Add axis for M_inf and create model grid

        t0 = 13.7  # Gyr
        x = t0/self.tau
        self.M_inf = self.M_star[np.newaxis, :] / (
                1 - np.exp(-x[:, np.newaxis])*(1+x[:, np.newaxis]))
        self.log_M_inf = np.log10(self.M_inf)
        
        delta_mag = 2.5 * np.log10(self.M_inf)

        self.u = np.array(u_model)[:, :, :, np.newaxis] - delta_mag[np.newaxis, :, np.newaxis, :]
        self.g = np.array(g_model)[:, :, :, np.newaxis] - delta_mag[np.newaxis, :, np.newaxis, :]
        self.r = np.array(r_model)[:, :, :, np.newaxis] - delta_mag[np.newaxis, :, np.newaxis, :]
        self.i = np.array(i_model)[:, :, :, np.newaxis] - delta_mag[np.newaxis, :, np.newaxis, :]
        self.z = np.array(z_model)[:, :, :, np.newaxis] - delta_mag[np.newaxis, :, np.newaxis,  :]
        
        print('PhotoModels loaded')
        
        
    def load_specmod(self, path, n_metallicities_interpolation=0):

        tau_model = []        
        extinction = []        
        lick_indices = []
        
        for Z_i in self.input_metallicities:  # TODO: Include IMF
            specmod = np.loadtxt(
                    path +
                    'lick_indices_Z_{:.4f}.txt'.format(Z_i),
                     unpack=True)
            
            tau = specmod[0, :]
            ext = specmod[1, :]
            lick = specmod[2:, :]
            
            
            tau_model.append(tau.reshape(300,30))
            extinction.append(ext.reshape(300,30))
            lick_indices.append(lick.reshape(lick.shape[0], 300,30))            
        
        self.tau = tau_model[-1][:, -1]
        self.extinction = extinction[-1][-1, :]
        self.log_tau = np.log10(self.tau)
        print('tau_min = {:.2f}, tau_max = {:.2f}'.format(
                min(self.tau), max(self.tau)))
        self.metallicities = self.input_metallicities
        self.log_Z = np.log10(self.metallicities)

        # Define range of metallicities
        if n_metallicities_interpolation > 0:
#            new_metallicities = np.linspace(self.input_log_Z[0], self.input_log_Z[-1],
#                                            n_metallicities_interpolation)
            new_metallicities = np.log10(np.linspace(
                    self.input_metallicities[0], 
                    self.input_metallicities[-1],
                    n_metallicities_interpolation)
                                        )
            
            lick_indices = interp1d(self.input_log_Z, lick_indices,
                                    axis=0)(new_metallicities)

            self.metallicities = 10**new_metallicities
            self.log_Z = new_metallicities
            print('New range of metallicities: \n', new_metallicities)


        self.lick_indices = lick_indices
        print('SpecModels loaded')
        
    def compute_V_max(self):
        u_limit = 19
        g_limit = 19
        r_limit = 17.77
        z_min = 0.02
        z_max = 0.07
        H0 = 70  # km/s/Mpc
        D_z_min = z_min * 3e5/H0
        D_z_max = z_max * 3e5/H0
        survey_solid_angle = 1317  # deg**2
        survey_solid_angle *= (np.pi/180)**2  # steradian

        D_u = 10**(0.2 * (u_limit-self.u) - 5)  # Mpc
        D_g = 10**(0.2 * (g_limit-self.g) - 5)  # Mpc
        D_r = 10**(0.2 * (r_limit-self.r) - 5)  # Mpc

        D_max = np.min([D_u, D_g, D_r], axis=(0)).clip(D_z_min, D_z_max)
        self.V_max = survey_solid_angle * (D_max**3 - D_z_min**3) / 3

    def compute_N_galaxies(self):
        """
        Compute number density of galaxies per unit comoving volume within
        each bin of the grid, according to the ansatz, as well as
        total number of galaxies observed by the SDSS, taking into account
        Vmax (depending on metallicity).
        """
        # Cosmic number density:
        self.n_ansatz = Ansatz.dn_dMdtau_grid(self.M_star, self.tau)
        d_log_tau = np.concatenate((
                np.array([self.log_tau[1]-self.log_tau[0]]),
                (self.log_tau[2:]-self.log_tau[:-2]) / 2,
                np.array([self.log_tau[-1]-self.log_tau[-2]])
                ))
        d_log_M = np.concatenate((
                np.array([self.log_M_star[1]-self.log_M_star[0]]),
                (self.log_M_star[2:]-self.log_M_star[:-2]) / 2,
                np.array([self.log_M_star[-1]-self.log_M_star[-2]])
                ))
        d_log_tau = np.abs(d_log_tau)
        self.n_ansatz *= self.tau[:, np.newaxis]*d_log_tau[:, np.newaxis]
        self.n_ansatz *= self.M_star[np.newaxis, :]*d_log_M[np.newaxis, :]
        self.n_ansatz *= np.log(10)**2

        # Observed by SDSS:
        self.N_galaxies = self.n_ansatz[np.newaxis, :, :] * self.V_max

    def bayesian_model_assignment(self, **kwargs):
        u = kwargs['u']
        g = kwargs['g']
        r = kwargs['r']
        i = kwargs['i']
        z = kwargs['z']
#        l_idx = kwargs['lick_indices']   # CAVEAT: Lick indices list must be 
                                         # equally ordered as model lick indices
        u_err = kwargs['u_err']
        g_err = kwargs['g_err']
        r_err = kwargs['r_err']
        i_err = kwargs['i_err']
        z_err = kwargs['z_err']
        
        lick_indices = kwargs['lick_indices']
        lick_indices_err = kwargs['lick_indices_err']
        
        Z_galaxy = []
        gal_extinct = []
        tau_galaxy = []
        M_inf_galaxy = []
        likelihood_all = []
        likelihood_lick_all = []
        self.min_err = 0.001
        min_err =self.min_err
        # TODO: Check number of given elements, include the possibility of varying the number of constrains

#        for galaxy in range(len(u)):
        for galaxy in range(0, 5):
            print('# Total % {:.2f}'.format(galaxy/len(u)*100))
            likelihood = (
                    ((u[galaxy]-self.u)**2)/(min_err+u_err[galaxy])**2 +(
                    (g[galaxy]-self.g)**2)/(min_err+g_err[galaxy])**2 + (
                    (r[galaxy]-self.r)**2)/(min_err+r_err[galaxy])**2+ (
                    (i[galaxy]-self.i)**2)/(min_err+i_err[galaxy])**2+(
                    (z[galaxy]-self.z)**2)/(min_err+z_err[galaxy])**2 
                          )
                    
            likelihood = likelihood/np.mean(likelihood)# [met, tau, ext, mass]
            
            # [met, tau, ]
            
            
            likelihood_lick =(lick_indices[np.newaxis, :, galaxy, np.newaxis, 
                                    np.newaxis]-self.lick_indices)**2\
                                /(lick_indices_err[np.newaxis, :, galaxy, 
                        np.newaxis, np.newaxis])**2
            
    
            # Collapsing over all lick indices --> [met, tau, ext]
            likelihood_lick = np.nansum(likelihood_lick, axis=1)                                                        
            likelihood_lick = likelihood_lick/np.nanmean(likelihood_lick)
#            ones = np.ones_like(likelihood)                                
#            likelihood_lick = likelihood_lick[:,:,:,np.newaxis]*ones
#            likelihood_lick = likelihood_lick/np.mean(likelihood_lick)
            
#            likelihood = likelihood[:,np.newaxis, :, : ,:]*np.ones_like(likelihood_lick)
            likelihood = likelihood+likelihood_lick[:,:,:, np.newaxis]
            likelihood_lick_all.append(likelihood_lick)
            del likelihood_lick
            
            
            likelihood = np.exp(- likelihood/2)
#            likelihood = np.exp(- likelihood)                        
            self.likelihood = likelihood
            
            dp_dlogtau = np.sum(likelihood, axis=(0, 2, 3))
            self.dp_dlogtau = dp_dlogtau
            log_tau_i = np.sum(self.log_tau*dp_dlogtau)/np.sum(dp_dlogtau)
            tau_galaxy.append(10**log_tau_i)
#            Model_grid.savefunction(self.log_tau, dp_dlogtau,
#                                   'tau_'+str(galaxy)+'.png')
            
            dp_dext = np.sum(likelihood, axis=(0, 1, 3))    
            gal_ext_i = np.sum(self.extinction*dp_dext)/np.sum(dp_dext)
            gal_extinct.append(gal_ext_i)
#            Model_grid.savefunction(self.extinction, dp_dext,
#                                    'ext_'+str(galaxy)+'.png')
            dp_dlogZ = np.sum(likelihood, axis=(1, 2, 3))
            log_Z_i = np.sum(self.log_Z*dp_dlogZ)/np.sum(dp_dlogZ)
            Z_galaxy.append(10**log_Z_i)
#            Model_grid.savefunction(self.log_Z, dp_dlogZ,
#                                    'Z_'+str(galaxy)+'.png')

            dp_dlogMinf = np.sum(likelihood, axis=(0, 1, 2))
            log_M_inf_i = np.sum(self.log_M_star*dp_dlogMinf)/np.sum(dp_dlogMinf)            
            M_inf_galaxy.append(10**log_M_inf_i)
            
            likelihood_all.append(likelihood)
#            Model_grid.savefunction(self.log_M_star, dp_dlogMinf,
#                                    'M_'+str(galaxy)+'.png')
            
        print('Bayesian assignment finished!, returned: Z_galaxy[], tau_galaxy[], M_inf_galaxy[]')

        return np.array(Z_galaxy), np.array(tau_galaxy), np.array(gal_extinct), np.array(M_inf_galaxy), likelihood_lick_all, likelihood_all

    def savefunction(x,y, name):
        
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(x,y)
        plt.savefig(name)
        plt.close()
        
        
