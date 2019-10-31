#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:14:10 2019

@author: pablo
"""

import numpy as np

import sys
sys.path.append("..")
sys.path.append("../products_codes")

from glob import glob

from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from dust_extinction_model_grid import DustModelGrid
from readspectrum import read_spectra

import pandas as pd
from matplotlib import pyplot as plt

# =============================================================================
# How to read spectra 
# =============================================================================

sdss_spectra = glob('../data/SDSS/spectra/*.fits')

# =============================================================================
# Model parameters
# =============================================================================

SEDfolder = '../population_synthesis/tau_delayedEXPSFR/epoch13.7Gyr/SED_kro*'
SEDpaths = np.sort(glob(SEDfolder))

metallicities = [
        '0.0001',
        '0.0004',
        '0.0040',
        '0.0080',
        '0.0200',
        '0.0500'
        ]

n_taus= 300
taus = np.logspace(-0.7, 1.7, n_taus)
SEDpaths = SEDpaths.reshape(len(metallicities), n_taus)

# DUST
ext_law ='calzetti2000'
#ext_law ='cardelli89'
n_dust_models = 30

SED_models = np.empty(
        (len(metallicities),
         n_taus,         
         1229
         ))

# =============================================================================
# Normalization window asn emission lines mask
# =============================================================================

def smooth_spectra(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def emission_line_mask(wavelength, width=20):
    """
    Emission lines masked:
        - [OII]λλ3726,3729,
        - [NeIII]λ3869, 
        -  Heps,Hδ,Hγ,Hβ,
        - [OIII]λλ4959,5007, 
        - [HeI]λ5876, 
        - NaDλ5890, 
        - [OI]λ6300, 
        - [NII]λλ6548,6583, 
        - Hα,
        - [SII]λλ6717,6731
    """
    emission_lines = [3726, 3729, 
                      3869,
                      3970, 4101, 4340, 4861, 
                      4959, 5007,
                      5876,
                      6300,
                      6548, 6583,
                      6563,
                      6717,
                      6717, 6731]
    
    mask = np.ones_like(wavelength)
    for line_i in emission_lines:
        mask[np.where(
                (wavelength>line_i-width)&(wavelength<line_i+width)
                )[0]] = 0
    return mask, emission_lines

###############################################################################
###############################################################################
###############################################################################

# =============================================================================
# Loading models
# =============================================================================

print('\nLoading set of model SEDs...\n')
for i_elem, Z_i in enumerate(metallicities):
    for j_elem in range(n_taus):
        flux = np.loadtxt(SEDpaths[i_elem, j_elem], usecols=4) #TODO: select which column to read
    
        
        SED_models[i_elem, j_elem, :] = flux
        
wavelength = np.loadtxt(SEDpaths[i_elem, j_elem], usecols=0) 
del flux, SEDpaths

# =============================================================================
# Metalliciticy interpolation
# =============================================================================
print('\nMetallicity interpolation of SEDs...\n')
metallicities = np.array(metallicities, dtype=float)
n_metallicities_interpolation = 20

new_metallicities = np.log10(np.logspace(
                    np.log10(metallicities[1]),
                    np.log10(metallicities[-1]),
                    n_metallicities_interpolation)
                                        )
            
SED_models = interp1d(np.log10(metallicities), SED_models, axis=0)(new_metallicities)
metallicities = new_metallicities
del new_metallicities, n_metallicities_interpolation
# =============================================================================
# Dust extinction grid            
# =============================================================================

SED_grid = np.empty(
        (len(metallicities),
         n_taus,
         n_dust_models,
         1229
         ))

print('\nBuilding Dust extinction grid...\n')
for i_elem, Z_i in enumerate(metallicities):
    for j_elem in range(n_taus):
        dustgrid = DustModelGrid(flux=SED_models[i_elem, j_elem, :],
                                 wavelength=wavelength, 
                                 ext_law_name=ext_law,
                                 dust_dimension = n_dust_models)

        fluxgrid = dustgrid.SED_grid    
        
        SED_grid[i_elem, j_elem, :, :] = fluxgrid.T

A_V = dustgrid.A_V
del SED_models, dustgrid

#%%
def lhood_plots(lhood, name, met, tau, Av, chimin, bayesian, bayes_err):
#    lhood = lhood/np.sum(lhood)
    met_tau_plane = np.mean(lhood, axis=2)
    met_ext_plane = np.mean(lhood, axis=1)
    tau_ext_plane = np.mean(lhood, axis=0)
    
    logtau = np.log10(tau)
    logZ = np.log10(met)
    logAv = np.log10(Av)
    
    bay_logZ = np.log10(bayesian[0])
    bay_logZerr = np.log10(bayes_err[0])
    bay_logtau = np.log10(bayesian[1])
    bay_logtauerr = np.log10(bayes_err[1])
    bay_logAv = np.log10(bayesian[2])
    bay_logAverr = np.log10(bayes_err[2])

    plt.figure(figsize=(13, 9))
    plt.subplot(231)
    plt.xlabel(r'$\log(\tau/$Gyr) ') 
    plt.ylabel(r'$\log$(Z)')
    plt.contourf(logtau, logZ, met_tau_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logtau[chimin[1]], logZ[chimin[0]], 'ro', 
                          markeredgecolor='k',
                          markersize=10)
    plt.errorbar(bay_logtau, bay_logZ, xerr=bay_logtauerr, yerr=bay_logZerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)    
    plt.grid(b=True)
    plt.colorbar()
    plt.subplot(232)
    plt.xlabel(r'$\log(A_V)$')    
    plt.ylabel(r'$\log$(Z)')    
    plt.contourf(logAv, logZ, met_ext_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logAv[chimin[2]], logZ[chimin[0]], 'ro',
                          markeredgecolor='k',
                          markersize=10)
    plt.errorbar(bay_logAv, bay_logZ, xerr=bay_logAverr, yerr=bay_logZerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10) 
    plt.grid(b=True)
    plt.colorbar()
    plt.subplot(233)
    plt.xlabel(r'$\log(A_V)$')
    plt.ylabel(r'$\log(\tau/[Gyr]$')
    plt.contourf(logAv, logtau, tau_ext_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logAv[chimin[2]], logtau[chimin[1]], 'ro', 
             markeredgecolor='k',
             markersize=10)
    plt.errorbar(bay_logAv, bay_logtau, xerr=bay_logAverr, yerr=bay_logtauerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10) 
    plt.colorbar()
    plt.grid(b=True)
    plt.subplot(234)
    plt.plot(logZ, np.sum(met_tau_plane, axis=1)/np.sum(met_tau_plane), 'k')
    plt.axvline(logZ[chimin[0]], color='red', label='Min chi2')
    plt.axvline(bay_logZ, color='lime', label='Bayesian')
    plt.axvspan(bay_logZ-bay_logZerr, bay_logZ+bay_logZerr, 
                color='lime', alpha=0.3)
    plt.legend()
    plt.xlabel('$\log(Z)$')
    plt.subplot(235)
    plt.plot(logtau, np.sum(met_tau_plane, axis=0)/np.sum(met_tau_plane), 'k')
    plt.axvline(logtau[chimin[1]], color='red')
    plt.axvline(bay_logtau, color='lime')
    plt.axvspan(bay_logtau-bay_logtauerr, bay_logtau+bay_logtauerr, 
                color='lime', alpha=0.3)
    plt.xlabel(r'$\log(\tau)$')
    plt.subplot(236)
    plt.plot(logAv, np.sum(tau_ext_plane, axis=0)/np.sum(tau_ext_plane), 'k')
    plt.axvline(logAv[chimin[2]], color='red')
    plt.axvline(bay_logAv, color='lime')
    plt.axvspan(bay_logAv-bay_logAverr, bay_logAv+bay_logAverr, 
                color='lime', alpha=0.3)
    plt.xlabel('$\log(Av)$')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(name+'_likelihoods.png')
    plt.close()
    
###############################################################################
###############################################################################
###############################################################################
#%%

chi2 = np.empty_like(SED_grid[:,:,:,0])
chi2_list = []
min_chi2 = np.zeros((len(sdss_spectra), 3), dtype=int)
bayesian_params = np.zeros((len(sdss_spectra), 3))
bayesian_errs = np.zeros((len(sdss_spectra), 3))

#for galaxy_i in range(len(sdss_spectra)):
#for galaxy_i in range(5):
for galaxy_i in [4]:
    print('galaxy #{}'.format(galaxy_i))
    
    spectra = read_spectra(sdss_spectra[galaxy_i], 'SDSS')
    gal_flux = spectra.flux
    wl = spectra.wavelength
    sigma = spectra.sigma
     
#    gal_flux_smth = smooth_spectra(gal_flux, 20)
    gal_flux_smth = gal_flux
  
    wavelength_pts = np.where((wavelength>=wl[0])&(wavelength<=wl[-1]))[0]
    reduced_wl = wavelength[wavelength_pts]
#    norm_pts = np.where((reduced_wl>=wl[-5]-300)&(reduced_wl<=wl[-5]))[0]
    norm_pts = np.where((reduced_wl>=wl[0])&(reduced_wl<=wl[-1]))[0]
    
    ## degrading and normalization of the observed flux ##
    gal_flux_smth = interp1d(wl, gal_flux_smth)(reduced_wl)
    norm_gal_flux = np.nansum(gal_flux_smth[norm_pts])
    gal_flux_smth = gal_flux_smth/norm_gal_flux
    sigma = interp1d(wl, sigma)(reduced_wl)
    sigma = sigma/norm_gal_flux
    ## computation of emission lines mask ##    
    emission_mask, emission_lines = emission_line_mask(reduced_wl)
    ## reduction of models to the observed range and normalization ##
    red_SED_grid = SED_grid[:,:,:, wavelength_pts]
    norm_flux = np.nansum(red_SED_grid[:,:,:, norm_pts], axis=3)
    red_SED_grid = red_SED_grid/norm_flux[:,:,:, np.newaxis]    
    ## computation of chi2 ##
    all_chi2 = (red_SED_grid[:,:,:,:]-gal_flux_smth[np.newaxis, np.newaxis, 
                  np.newaxis, :])**2\
                  /sigma[np.newaxis, np.newaxis, np.newaxis, :]**2\
                  *emission_mask[np.newaxis, np.newaxis, np.newaxis, :]
    chi2 = np.nanmedian(all_chi2, axis=3)    
#    chi2 = np.nansum(all_chi2*np.diff(reduced_wl)[0], axis=3)/2000    
#    chi2_mean = np.nanmean(all_chi2, axis=3)          
    min_chi2[galaxy_i, :]=np.where(chi2==np.min(chi2))

#    chi2_list.append(chi2)
    
    lhood = np.exp(-chi2/2)

    ## Computation of params mean value and std ##
    
    dpdlogZ = np.nansum(lhood, axis=(1,2))
    logZ_gal = np.sum(dpdlogZ*metallicities)/np.sum(dpdlogZ)
    Z_gal = 10**logZ_gal
    std_logZ_gal = np.sum(
            (metallicities-logZ_gal)**2*dpdlogZ
            )/np.sum(dpdlogZ)
    std_logZ_gal = np.sqrt(std_logZ_gal)
    
    
    dpdlogtau = np.nansum(lhood, axis=(0,2))
    logtau_gal = np.sum(dpdlogtau*np.log10(taus))/np.sum(dpdlogtau)
    tau_gal = 10**logtau_gal
    std_logtau_gal = np.sum(
            (np.log10(taus)-logtau_gal)**2 *dpdlogtau
            )/np.sum(dpdlogtau)
    std_logtau_gal = np.sqrt(std_logtau_gal)    
    
    dpdAv = np.nansum(lhood, axis=(0,1))
    logAv_gal = np.sum(dpdAv*np.log10(A_V))/np.sum(dpdAv) 
    Av_gal = 10**logAv_gal
    std_logAv_gal = np.sum(
            (np.log10(A_V)-logAv_gal)**2*dpdAv
            )/np.sum(dpdAv)
    std_logAv_gal = np.sqrt(std_logAv_gal)
    
    dlogZ = np.diff(metallicities)[0]
    dlogtau = np.diff(np.log10(taus))[0]
    dlogAv = np.diff(np.log10(A_V))[0]    
    
    plogzlogtaulogav = 1/(
            metallicities[-1]-metallicities[0]+\
            np.log10(taus[-1]/taus[0])+np.log10(A_V[-1]/A_V[0]))
    model_evidence = np.sum(plogzlogtaulogav*lhood*dlogZ*dlogtau*dlogAv)
    
    print(model_evidence)
    
    bayesian_params[galaxy_i, :] = [Z_gal, tau_gal, Av_gal]
    bayesian_errs[galaxy_i, :] = [std_logZ_gal, std_logtau_gal, std_logAv_gal]


# =============================================================================
#   PLOTS
# =============================================================================
    lhood_plots(lhood, 'likelihoods/'+str(galaxy_i), 
                10**metallicities,
                taus,
                A_V,
                min_chi2[galaxy_i, :],
                bayesian_params[galaxy_i, :],
                bayesian_errs[galaxy_i, :]
                )
          
    print(' Bayesian: Z={:.3}, tau={:.2}, Av={:.2}'.format(Z_gal, tau_gal, Av_gal))
    print(' Min chi2: Z={:.3}, tau={:.2}, Av={:.2}'.format(
                       10**metallicities[min_chi2[galaxy_i,0]],
                       taus[min_chi2[galaxy_i,1]],
                       A_V[min_chi2[galaxy_i,2]]))
    # -----------------------------------------
    ## bayesian model interpolation ##
    grid_interp = RegularGridInterpolator(
            (10**metallicities, taus, A_V, reduced_wl), 
            red_SED_grid)
    f_bayes = grid_interp((bayesian_params[galaxy_i,0], 
                              bayesian_params[galaxy_i,1],
                              bayesian_params[galaxy_i,2],
                              reduced_wl))
    
    minchi2_Z = min_chi2[galaxy_i,0]
    minschi2_tau = min_chi2[galaxy_i,1]
    minchi2_Av = min_chi2[galaxy_i,2]
    f  = red_SED_grid[minchi2_Z, minschi2_tau, minchi2_Av, :]
    plt.rcParams['axes.facecolor'] = 'black'
    plt.figure(figsize=(10,9))
    plt.subplot(211)
    plt.semilogy(reduced_wl, f,
               'r', 
               label=r'min $\chi^2$: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                       10**metallicities[min_chi2[galaxy_i,0]],
                       taus[min_chi2[galaxy_i,1]],
                       A_V[min_chi2[galaxy_i,2]])
               )
    plt.semilogy(reduced_wl, f_bayes, color='lime', 
                 label= r'Bayesian: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                              bayesian_params[galaxy_i,0], 
                              bayesian_params[galaxy_i,1],
                              bayesian_params[galaxy_i,2])
                 )
                 
    plt.semilogy(reduced_wl, gal_flux_smth, 'w', label='Measured spectra')
    for em_line in emission_lines:
        plt.axvspan(em_line-10, em_line+10, color='cyan', alpha=0.4)
    plt.legend(facecolor='white', framealpha=0.9)
#    plt.grid(b=True, color='white')
    plt.subplot(212)
    plt.semilogy(reduced_wl, np.abs(f-gal_flux_smth)/gal_flux_smth, 'r')
    plt.semilogy(reduced_wl, np.abs(f_bayes-gal_flux_smth)/gal_flux_smth, 
                 color='lime')
    plt.axhline(np.nanmedian(np.abs(f-gal_flux_smth)/gal_flux_smth), 
                linestyle='--', color='red', label=r'$\chi^2$ median error')
    plt.axhline(np.nanmedian(np.abs(f_bayes-gal_flux_smth)/gal_flux_smth),
                linestyle='--', color='lime', label=r'Bayesian median error')
    plt.ylabel(r'$\frac{|M-Obs|}{Obs}$')
    plt.grid(b=True, color='white')
    plt.legend(facecolor='white', framealpha=0.9)
    plt.savefig('spectra_fits/spectra_testfit'+str(galaxy_i)+'.png')
    plt.close()
    plt.rcParams['axes.facecolor'] = 'white'


    del f         

# =============================================================================
#  OUTPUT DATA        
# =============================================================================
print('No saving data')
#data = {'Galaxy':sdss_spectra, 
#        'met_mod_pos':min_chi2[:,0],
#        'Z':10**metallicities[min_chi2[:,0]],
#        'tau_mod_pos':min_chi2[:,1],
#        'tau':taus[min_chi2[:,1]],
#        'extinct_mod_pos':min_chi2[:,2],
#        'ext_Av':A_V[min_chi2[:,2]]
#        }    
#df = pd.DataFrame(data=data)    
#df.to_csv('minchi2_specfittprod.csv')
#
#bayesian_data = {'Galaxy':sdss_spectra,
#                 'met_Z': bayesian_params[:,0],
#                 'tau': bayesian_params[:,1],
#                 'ext_Av': bayesian_params[:,2],
#                 }
#df = pd.DataFrame(data=bayesian_data)
#df.to_csv('bayesian_specfittprod.csv')
#
#
#del data, df, bayesian_data






