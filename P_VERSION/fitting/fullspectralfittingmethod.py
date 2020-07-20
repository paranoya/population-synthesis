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
import os

from glob import glob

from scipy.interpolate import interp1d, RegularGridInterpolator
from dust_extinction_model_grid import DustModelGrid
from readspectrum import read_spectra
from expectation_maximization import expectation_maximization
from tophat import tophat

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# How to read spectra & sample photometry
# =============================================================================

## The photometry table is ordered in terms of plate, mjd and fiber ID in the
## same way as is ordenered within the variable containing the spectrum paths

sdss_spectra = np.sort(glob('../data/SDSS/sample1/spectra/*.fits'))

df = pd.read_csv('../data/SDSS/sample1/sample1_photometry.csv')
photometry = np.array([df['u_abs'], df['g_abs'], 
                       df['r_abs'], df['i_abs'], df['z_abs']])
sigma_photo = np.array([df['u_abserr'], df['g_abserr'], 
                       df['r_abserr'], df['i_abserr'], df['z_abserr']])

del df
# =============================================================================
# Outputs
# =============================================================================

output_path = '../results/test/sample1'
likelihoods_path = output_path+'/likelihoods'
specfits_path = output_path+'/spectrum_fits'
photo_path = output_path+'/photometry_fits'

if os.path.isdir(output_path)==False:
    os.mkdir(output_path)
    os.mkdir(likelihoods_path)
    os.mkdir(specfits_path)    
    os.mkdir(photo_path)    
    print('\nNew folders created at', output_path)
    
# =============================================================================
# Model parameters
# =============================================================================

SEDfolder = '../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr/SED_cha*'
SEDpaths = np.sort(glob(SEDfolder))

PHOTOfolder = '../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr/'+\
                'products/photometry*'
PHOTOpaths = np.sort(glob(PHOTOfolder))
metallicities = [
        '0.0001',
        '0.0004',
        '0.0040',
        '0.0080',
        '0.0200',
        '0.0500'
        ]

n_taus= 50
taus = np.logspace(-1, 1.7, n_taus)
dlogtau = np.diff(np.log10(taus))[2]
SEDpaths = SEDpaths.reshape(len(metallicities), n_taus)

n_wave = 6900
# DUST
ext_law ='calzetti2000'
#ext_law ='cardelli89'
n_dust_models = 30

SED_models = np.empty(
        (len(metallicities),
         n_taus,         
         6900
         ))

# =============================================================================
# Normalization window and emission lines mask
# =============================================================================

def smooth_spectra(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class emission_line_mask(object):
    
    def __init__(self, wavelength, width=50):
        self.wavelength=wavelength
        self.width = width
        self.mask = np.ones_like(self.wavelength)
        
        self.emission_lines_list()
        list(map(self.mask_line, self.emission_lines))

    
    def mask_line(self, emission_line):        
        self.mask[np.where(
                (self.wavelength>emission_line-self.width)&(
                 self.wavelength<emission_line+self.width)
                )[0]] = 0
    def emission_lines_list(self):    
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
        self.emission_lines = [3726, 3729, 
                      3869,
                      3970, 4101, 4340, 4861, 
                      4959, 5007,
                      5876,
                      6300,
                      6548, 6583,
                      6563,
                      6717,
                      6717, 6731]
    
    

###############################################################################
###############################################################################
###############################################################################

# =============================================================================
# Loading models
# =============================================================================

print('\nLoading set of model SEDs from: \n', SEDfolder)
for i_elem, Z_i in enumerate(metallicities):
    for j_elem in range(n_taus):
        flux = np.loadtxt(SEDpaths[i_elem, j_elem], usecols=4) #TODO: select which column to read
    
        
        SED_models[i_elem, j_elem, :] = flux
        
wavelength = np.loadtxt(SEDpaths[i_elem, j_elem], usecols=0) 
del flux, SEDpaths

# =============================================================================
# Photometry models
# =============================================================================

#PHOTO_models = np.empty((5, len(metallicities), len(taus), n_dust_models), 
#                        dtype=np.float32)
#for i_elem in range(len(metallicities)):
#    
#    mag = np.loadtxt(PHOTOpaths[i_elem], usecols=(2,3,4,5,6), unpack=True)            
#  
#    PHOTO_models[:, i_elem, :, :] = mag.reshape(5, len(taus), n_dust_models)
#        
#del mag, i_elem


# =============================================================================
# Metalliciticy interpolation
# =============================================================================
print('\nModel Metallicity interpolation...\n')
metallicities = np.array(metallicities, dtype=float)
n_metallicities_interpolation = 20

new_metallicities = np.log10(np.logspace(
                    np.log10(metallicities[1]),
                    np.log10(metallicities[-1]),
                    n_metallicities_interpolation)
                                        )
            
SED_models = interp1d(np.log10(metallicities), SED_models, axis=0)(new_metallicities)
#PHOTO_models = interp1d(np.log10(metallicities), PHOTO_models, axis=1)(new_metallicities)
metallicities = new_metallicities
dlogZ = np.diff(metallicities)[2]

del new_metallicities, n_metallicities_interpolation

masses = np.logspace(6.5, 12.5, 100)
logmasses = np.log10(masses)
dlogmass = np.diff(np.log10(masses))[2]
delta_mag = 2.5*np.log10(masses)

#PHOTO_models = PHOTO_models[:,:,:,:, np.newaxis]-delta_mag[np.newaxis,
#                                        np.newaxis, np.newaxis, np.newaxis, :]
#PHOTO_models = np.float32(PHOTO_models)
del delta_mag

# =============================================================================
# Dust extinction grid            
# =============================================================================

SED_grid = np.empty(
        (len(metallicities),
         n_taus,
         n_dust_models,
         6900
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
dlogAv = np.diff(np.log10(A_V))[2]
del SED_models, dustgrid


#%%
def lhood_plots(lhood, name, met, tau, Av, chimin, bayesian, bayes_err, 
                two_sol, two_solerr):
#    lhood = lhood/np.sum(lhood)
    met_tau_plane = np.mean(lhood, axis=2)
    met_ext_plane = np.mean(lhood, axis=1)
    tau_ext_plane = np.mean(lhood, axis=0)
    
    sol1 = two_sol[0]
    sol2 = two_sol[1]
    sol1err = two_solerr[0]
    sol2err = two_solerr[1]    
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
    # -------------------------------------------------------------------------
    plt.subplot(231)
    plt.xlabel(r'$\log(\tau/$Gyr) ') 
    plt.ylabel(r'$\log$(Z)')
    plt.contourf(logtau, logZ, met_tau_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logtau[chimin[1]], logZ[chimin[0]], '^', color='fuchsia', 
                          markeredgecolor='k',
                          markersize=10)
    plt.errorbar(bay_logtau, bay_logZ, xerr=bay_logtauerr, yerr=bay_logZerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)    
    plt.errorbar(sol1[1], sol1[0], xerr=sol1err[1], yerr=sol1err[0],
             fmt='o', color='red', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.errorbar(sol2[1], sol2[0], xerr=sol2err[1], yerr=sol2err[0],
             fmt='o', color='cyan', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.xlim(logtau[0], logtau[-1])
    plt.ylim(logZ[0], logZ[-1])
    plt.grid(b=True)
    plt.colorbar()
    # -------------------------------------------------------------------------
    plt.subplot(232)
    plt.xlabel(r'$\log(A_V)$')    
    plt.ylabel(r'$\log$(Z)')    
    plt.contourf(logAv, logZ, met_ext_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logAv[chimin[2]], logZ[chimin[0]],  '^', color='fuchsia',
                          markeredgecolor='k',
                          markersize=10)
    plt.errorbar(bay_logAv, bay_logZ, xerr=bay_logAverr, yerr=bay_logZerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10) 
    plt.errorbar(sol1[2], sol1[0], xerr=sol1err[2], yerr=sol1err[0],
             fmt='o', color='red', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.errorbar(sol2[2], sol2[0], xerr=sol2err[2], yerr=sol2err[0],
             fmt='o', color='cyan', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.xlim(logAv[0], logAv[-1])
    plt.ylim(logZ[0], logZ[-1])
    plt.grid(b=True)
    plt.colorbar()
    # -------------------------------------------------------------------------
    plt.subplot(233)
    plt.xlabel(r'$\log(A_V)$')
    plt.ylabel(r'$\log(\tau/[Gyr]$')
    plt.contourf(logAv, logtau, tau_ext_plane, origin='lower', cmap='YlGnBu',
               vmin=0, vmax=0.5, levels=40
               )
    plt.plot(logAv[chimin[2]], logtau[chimin[1]], '^', color='fuchsia',
             markeredgecolor='k',
             markersize=10)
    plt.errorbar(bay_logAv, bay_logtau, xerr=bay_logAverr, yerr=bay_logtauerr,
             fmt='o', color='lime', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10) 
    plt.errorbar(sol1[2], sol1[1], xerr=sol1err[2], yerr=sol1err[1],
             fmt='o', color='red', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.errorbar(sol2[2], sol2[1], xerr=sol2err[2], yerr=sol2err[1],
             fmt='o', color='cyan', 
             lolims=True, uplims=True, xlolims=True, xuplims=True,
             markeredgecolor='k',
             markersize=10)
    plt.xlim(logAv[0], logAv[-1])
    plt.ylim(logtau[0], logtau[-1])
    plt.colorbar()
    plt.grid(b=True)
    # -------------------------------------------------------------------------
    plt.subplot(234)
    plt.plot(logZ, np.sum(met_tau_plane, axis=1)/np.sum(met_tau_plane), 'k')
    plt.axvline(logZ[chimin[0]], color='fuchsia', label='Min chi2')
    plt.axvline(bay_logZ, color='lime', label='Bayesian')
    plt.axvspan(bay_logZ-bay_logZerr, bay_logZ+bay_logZerr, 
                color='lime', alpha=0.3)    
    plt.axvline(sol1[0], color='red', alpha=1, label='sol1')
    plt.axvspan(sol1[0]-sol1err[0], sol1[0]+sol1err[0], 
                color='red', alpha=0.3)
    plt.axvline(sol2[0], color='deepskyblue', alpha=1, label='sol2')
    plt.axvspan(sol2[0]-sol2err[0], sol2[0]+sol2err[0], 
                color='deepskyblue', alpha=0.3)
    plt.legend()
    plt.xlabel('$\log(Z)$')
    # -------------------------------------------------------------------------
    plt.subplot(235)
    plt.plot(logtau, np.sum(met_tau_plane, axis=0)/np.sum(met_tau_plane), 'k')
    plt.axvline(logtau[chimin[1]], color='fuchsia')
    plt.axvline(bay_logtau, color='lime')
    plt.axvspan(bay_logtau-bay_logtauerr, bay_logtau+bay_logtauerr, 
                color='lime', alpha=0.3)
    
    plt.axvline(sol1[1], color='red', alpha=1, label='sol1')
    plt.axvspan(sol1[1]-sol1err[1], sol1[1]+sol1err[1], 
                color='red', alpha=0.3)
    plt.axvline(sol2[1], color='deepskyblue', alpha=1, label='sol2')
    plt.axvspan(sol2[1]-sol2err[1], sol2[1]+sol2err[1], 
                color='deepskyblue', alpha=0.3)
    
    plt.xlabel(r'$\log(\tau)$')
    # -------------------------------------------------------------------------
    plt.subplot(236)
    plt.plot(logAv, np.sum(tau_ext_plane, axis=0)/np.sum(tau_ext_plane), 'k')
    plt.axvline(logAv[chimin[2]], color='fuchsia')
    plt.axvline(bay_logAv, color='lime')
    plt.axvspan(bay_logAv-bay_logAverr, bay_logAv+bay_logAverr, 
                color='lime', alpha=0.3)
    plt.axvline(sol1[2], color='red', alpha=1, label='sol1')
    plt.axvspan(sol1[2]-sol1err[2], sol1[2]+sol1err[2], 
                color='red', alpha=0.3)
    plt.axvline(sol2[2], color='deepskyblue', alpha=13, label='sol2')
    plt.axvspan(sol2[2]-sol2err[2], sol2[2]+sol2err[2], 
                color='deepskyblue', alpha=0.3)
   
    plt.xlabel('$\log(Av)$')
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    plt.savefig(name+'_likelihoods.png')
    plt.close()
    
###############################################################################
#########################    Spectrum fitting    ##############################
###############################################################################
#%%

chi2 = np.empty_like(SED_grid[:,:,:,0])
min_chi2 = np.zeros((len(sdss_spectra), 3), dtype=int)
bayesian_params = np.zeros((len(sdss_spectra), 3))
twosols_params = np.zeros((len(sdss_spectra), 6))
bayesian_errs = np.zeros((len(sdss_spectra), 3))
twosols_evidence = np.zeros((len(sdss_spectra), 2))

twosols_errs = np.zeros((len(sdss_spectra), 6))

model_evidence = np.zeros((len(sdss_spectra)))
fluxobjid = np.zeros((len(sdss_spectra)))

bayes_mass = np.zeros((len(sdss_spectra), 2), dtype=np.float16)

print('\nStarting spectrum fit...\n')
all_lhood = np.zeros_like(chi2)
dp_dmdtau_all = 0
#for galaxy_i in range(len(sdss_spectra)):
for galaxy_i in range(100):
#for galaxy_i in [11]:
    print('----------------------------')
    print('galaxy #{}'.format(galaxy_i))
    ## ---- PHOTOMETRY / MASS INFERENCE ---
#    gal_photo = photometry[:, galaxy_i]
#    gal_sigma_photo = sigma_photo[:, galaxy_i]
#    
#    photo_chi2 = (PHOTO_models -\
#            gal_photo[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])**2 \
#         /gal_sigma_photo[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]**2
#    photo_chi2 = np.nanmean(photo_chi2, axis=0)
#    photo_lhood = np.exp(-photo_chi2/2)
#    dpdlogtaudlogm  = np.sum(photo_lhood*dlogZ*dlogAv, axis=(0,2))\
#                            /np.sum(photo_lhood*dlogZ*dlogAv*dlogtau*dlogmass)
#    dpdlogm = np.sum(dpdlogtaudlogm*dlogtau, axis=0)\
#                            /np.sum(dpdlogtaudlogm*dlogtau*dlogmass)
#    dpdlogtau_photo = np.sum(dpdlogtaudlogm*dlogtau, axis=1)\
#                            /np.sum(dpdlogtaudlogm*dlogtau*dlogmass)
#    logm = np.sum(dpdlogm*logmasses*dlogmass)
#    sigma_logm = np.sum(dpdlogm*(logmasses-logm)**2*dlogmass)                             
#    
#    bayes_mass[galaxy_i, 0]  = logm
#    bayes_mass[galaxy_i, 1]  = sigma_logm
#    
#    plt.figure()
#    plt.contourf(logmasses, np.log10(taus), dpdlogtaudlogm , origin='lower',
#                 cmap='YlGnBu',
#                 levels=40
#                 )
#    plt.axvspan(logm-sigma_logm, logm+sigma_logm, color='lime', alpha=0.3) 
#    plt.colorbar()
#    plt.xlabel(r'$\log(M [M_\odot])$')
#    plt.ylabel(r'$\log(\tau [Gyr])$')
#    plt.savefig(photo_path+'/dpdlogtaudlogm_'+str(galaxy_i)+'.png')
#    plt.close()
#    del photo_chi2, photo_lhood, dpdlogtaudlogm #, dpdlogm, logm, sigma_logm
    
    ## ---- Read spectrum from fits files ---- ##
    spectra = read_spectra(sdss_spectra[galaxy_i], 'SDSS')
    spectra.get_SDSSinfo()
    fluxobjid[galaxy_i]= spectra.fluxobjid
    gal_flux = spectra.flux
    wl = spectra.wavelength
    sigma = spectra.sigma
    
    spectra.flux_to_luminosity()
    luminosity = spectra.luminosity
    del spectra
    
    ## ---- Spectra smoothing ---- ##     
#    gal_flux_smth = smooth_spectra(gal_flux, 20)
    gal_flux_smth = gal_flux
  
    ## ---- reducing resolution and normalization ---- ##
    wavelength_pts = np.where((wavelength>=wl[0])&(wavelength<=wl[-1]))[0]
    reduced_wl = wavelength[wavelength_pts]
    norm_pts = np.where((reduced_wl>=wl[0])&(reduced_wl<=wl[-1]))[0]    
    
    gal_flux_smth = interp1d(wl, gal_flux_smth)(reduced_wl)
    norm_gal_flux = np.nansum(gal_flux_smth[norm_pts])
    gal_flux_smth = gal_flux_smth/norm_gal_flux
    sigma = interp1d(wl, sigma)(reduced_wl)
    sigma = sigma/norm_gal_flux
    
    gal_luminosity = interp1d(wl, luminosity)(reduced_wl)

    mass_grid = np.nanmean(gal_luminosity[np.newaxis, np.newaxis, np.newaxis, :]\
                /SED_grid[:,:,:, wavelength_pts], axis=-1)
    mass_grid = np.log10(mass_grid)
    red_SED_grid = SED_grid[:,:,:, wavelength_pts]
    norm_flux = np.nansum(red_SED_grid[:,:,:, norm_pts], axis=3)
    red_SED_grid = red_SED_grid/norm_flux[:,:,:, np.newaxis]    

    
    
    ## ---- Computation of emission lines mask ---- ##    
    masquerading = emission_line_mask(reduced_wl)
    emission_mask= masquerading.mask
    emission_lines = masquerading.emission_lines
    
    ## ---- computation of chi2 ---- ##
    all_chi2 = (red_SED_grid[:,:,:,:]-gal_flux_smth[np.newaxis, np.newaxis, 
                  np.newaxis, :])**2\
                  /sigma[np.newaxis, np.newaxis, np.newaxis, :]**2\
                  *emission_mask[np.newaxis, np.newaxis, np.newaxis, :]
    chi2 = np.nanmean(all_chi2, axis=3)    
#    chi2_mean = np.nanmean(all_chi2, axis=3)          
    
    ## ---- Best fit == min(chi2) ---- ##
    min_chi2[galaxy_i, :]=np.where(chi2==np.min(chi2))

    ## ---- Likelihood and bayesiand results (mu, sigma) --- ##
    lhood = np.exp(-chi2/2)
    all_lhood = all_lhood+lhood
    
    two_sol = expectation_maximization(prob_distrib=lhood, x=metallicities,
                             y=np.log10(taus), z=np.log10(A_V))
    twosols_params[galaxy_i, :] = [two_sol.mu0[0], two_sol.mu0[1],
                                   two_sol.mu0[2], two_sol.mu1[0], 
                                   two_sol.mu1[1], two_sol.mu1[2]]    
    
    twosols_errs[galaxy_i, :] = [two_sol.sigma0[0],two_sol.sigma0[1],
                                 two_sol.sigma0[2],two_sol.sigma1[0],
                                 two_sol.sigma1[1],two_sol.sigma1[2]]
    twosols_evidence[galaxy_i, :]= [two_sol.ev0, two_sol.ev1]
    ## ---------------------------------------------------- ##
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
    
    bayesian_params[galaxy_i, :] = [Z_gal, tau_gal, Av_gal]
    bayesian_errs[galaxy_i, :] = [std_logZ_gal, std_logtau_gal, std_logAv_gal]
    
    mass = np.sum(mass_grid*lhood)/np.sum(lhood)
    sigma_mass = np.sum((mass_grid-mass)**2*lhood)/np.sum(lhood)
    bayes_mass[galaxy_i, 0] = mass
    bayes_mass[galaxy_i, 1] = sigma_mass
    

    ## ---- Model evidence ---- ##
    plogzlogtaulogav = 1/(
            metallicities[-1]-metallicities[0]+\
            np.log10(taus[-1]/taus[0])+np.log10(A_V[-1]/A_V[0]))
    
    evidence = np.sum(plogzlogtaulogav*lhood*dlogZ*dlogtau*dlogAv)
    model_evidence[galaxy_i] = evidence
    
    all_lhood = all_lhood+lhood*model_evidence[galaxy_i]
    
    
    print('Model evidence:', model_evidence[galaxy_i])
    
    masses = np.ones((50, 2))
    masses = masses*np.linspace(6.5, 12, 50)[:, np.newaxis]
    logm = np.linspace(6.5, 12, 50)
    delta_masses = np.diff(masses[:,0])[0]
    masses[:,0] = masses[:,0]-delta_masses/2
    masses[:,1] = masses[:,1]+delta_masses/2
        
    
    mass_filter = list(map(tophat, [mass_grid]*len(masses), masses.tolist()))
    mass_filter = np.array(mass_filter)
    
#    dpdmdtau = np.nansum(lhood[np.newaxis, :,:,:]*mass_filter*dlogtau*dlogAv, 
#                  axis=-1)
    dpdmdtau = np.nansum(lhood[np.newaxis, :,:,:]*mass_filter, 
                  axis=(1,3))/np.sum(lhood)
#    dpdm = np.nansum(lhood[np.newaxis, :,:,:]*mass_filter*
#                     dlogZ*dlogtau*dlogAv/evidence, 
#                  axis=(1,2,3))
    dp_dmdtau_all = dp_dmdtau_all + dpdmdtau
    


# =============================================================================
#   PLOTS
# =============================================================================
    lhood_plots(lhood, likelihoods_path+'/'+str(galaxy_i), 
                10**metallicities,
                taus,
                A_V,
                min_chi2[galaxy_i, :],
                bayesian_params[galaxy_i, :],
                bayesian_errs[galaxy_i, :],
                [two_sol.mu0,two_sol.mu1],
                [two_sol.sigma0,two_sol.sigma1] 
                )
          
    print(' Bayesian: Z={:.3}, tau={:.2}, Av={:.2}'.format(Z_gal, tau_gal, Av_gal))
    print(' Min chi2: Z={:.3}, tau={:.2}, Av={:.2}'.format(
                       10**metallicities[min_chi2[galaxy_i,0]],
                       taus[min_chi2[galaxy_i,1]],
                       A_V[min_chi2[galaxy_i,2]]))

    ## ---- Bayesian model interpolation ---- ##
    grid_interp = RegularGridInterpolator(
            (10**metallicities, taus, A_V, reduced_wl), 
            red_SED_grid)
    
    f_bayes = grid_interp((bayesian_params[galaxy_i,0], 
                              bayesian_params[galaxy_i,1],
                              bayesian_params[galaxy_i,2],
                              reduced_wl))
    f_twosol0 = grid_interp((10**two_sol.mu0[0],
                             10**two_sol.mu0[1],
                             10**two_sol.mu0[2],
                              reduced_wl))
    f_twosol1 = grid_interp((10**two_sol.mu1[0],
                             10**two_sol.mu1[1],
                             10**two_sol.mu1[2],
                              reduced_wl))
    minchi2_Z = min_chi2[galaxy_i,0]
    minschi2_tau = min_chi2[galaxy_i,1]
    minchi2_Av = min_chi2[galaxy_i,2]
    f  = red_SED_grid[minchi2_Z, minschi2_tau, minchi2_Av, :]
    
    plt.rcParams['axes.facecolor'] = 'black'
    plt.figure(figsize=(10,9))
    plt.subplot(211)
#    plt.errorbar(x=reduced_wl, y=gal_flux_smth, yerr=sigma, 
#                 color='w',ecolor='gray', alpha=1, label='Measured spectra'
#                 )
    plt.plot(reduced_wl, gal_flux_smth, color='w')
    plt.plot(reduced_wl, f,
               color='fuchsia', 
               label=r'min $\chi^2$: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                       10**metallicities[min_chi2[galaxy_i,0]],
                       taus[min_chi2[galaxy_i,1]],
                       A_V[min_chi2[galaxy_i,2]])
               )
    plt.plot(reduced_wl, f_bayes, color='lime', 
                 label= r'Bayesian: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                              bayesian_params[galaxy_i,0], 
                              bayesian_params[galaxy_i,1],
                              bayesian_params[galaxy_i,2])
                 )
    plt.plot(reduced_wl, f_twosol0, color='red', 
                 label= r'Sol0: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                             10**two_sol.mu0[0],
                             10**two_sol.mu0[1],
                             10**two_sol.mu0[2])
                 )
    plt.plot(reduced_wl, f_twosol1, color='deepskyblue', 
                 label= r'Sol1: Z={:.3}, $\tau$={:.2}, $A_V$={:.2}'.format(
                             10**two_sol.mu1[0],
                             10**two_sol.mu1[1],
                             10**two_sol.mu1[2])
                 )
    
                 
#    plt.plot(reduced_wl, gal_flux_smth, 'w', label='Measured spectra')
    plt.yscale('log')
    for em_line in emission_lines:
        if em_line==emission_lines[-1]:
            plt.axvspan(em_line-masquerading.width, em_line+masquerading.width,
                    color='cyan', alpha=0.4, label='masked bands')
        else:
            plt.axvspan(em_line-masquerading.width, em_line+masquerading.width,
                    color='cyan', alpha=0.4)
        
    plt.legend(facecolor='white', framealpha=0.9)
#    plt.grid(b=True, color='white')
    plt.subplot(212)
    chi = (f-gal_flux_smth)/sigma
    chi_bayes = (f_bayes-gal_flux_smth)/sigma
    plt.plot(reduced_wl, chi, 'r')
    plt.plot(reduced_wl, chi_bayes, 
                 color='lime')
    plt.axhline(np.nanmedian(chi), 
                linestyle='--', color='red', label=r'median $\chi$')
    plt.axhline(np.nanmedian(chi_bayes),
                linestyle='--', color='lime', label=r'median $\chi$ Bayesian fit')
    plt.ylabel(r'$\chi$')
    plt.yscale('symlog')
#    plt.ylim(-1,1)
    plt.grid(b=True, color='white')
    plt.legend(facecolor='white', framealpha=0.9)
    plt.savefig(specfits_path+'/spectra_testfit'+str(galaxy_i)+'.png')
    plt.close()
    plt.rcParams['axes.facecolor'] = 'white'
    
             

#%%
met_tau_plane = np.mean(all_lhood, axis=2)/len(sdss_spectra)
tau_dust_plane = np.mean(all_lhood, axis=0)/len(sdss_spectra)
met_dust_plane = np.mean(all_lhood, axis=1)/len(sdss_spectra)

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.xlabel(r'$\log(\tau/$Gyr) ') 
plt.ylabel(r'$\log$(Z)')
plt.contourf(np.log10(taus), metallicities, met_tau_plane, 
             origin='lower', cmap='YlGnBu',
             levels=40
               )
plt.colorbar()
plt.subplot(222)
plt.xlabel(r'$\log(A_V/$mag)') 
plt.ylabel(r'$\log$(Z)')
plt.contourf(np.log10(A_V), metallicities, met_dust_plane, 
             origin='lower', cmap='YlGnBu',
             levels=40
               )
plt.colorbar()
plt.subplot(224)
plt.xlabel(r'$\log(A_V/$mag)') 
plt.ylabel(r'$\log(\tau/$Gyr) ')
plt.contourf(np.log10(A_V), np.log10(taus), tau_dust_plane, 
             origin='lower', cmap='YlGnBu',
             levels=40
               )
plt.colorbar()
plt.subplot(223)
plt.xlabel(r'$\log(\tau/$Gyr) ') 
plt.ylabel(r'$\log(M_\infty/M_\odot)$')
plt.contourf(np.log10(taus), logm, dp_dmdtau_all, 
             origin='lower', cmap='YlGnBu',
             levels=40
               )
plt.colorbar()
plt.savefig(output_path+'/all_distrib.png')

# =============================================================================
#  OUTPUT DATA        
# =============================================================================
#print('No saving data')
print('\nSaving data...\n')
data = {'Galaxy':sdss_spectra, 
        'met_mod_pos':min_chi2[:,0],
        'Z':10**metallicities[min_chi2[:,0]],
        'tau_mod_pos':min_chi2[:,1],
        'tau':taus[min_chi2[:,1]],
        'extinct_mod_pos':min_chi2[:,2],
        'ext_Av':A_V[min_chi2[:,2]]
        }    
df = pd.DataFrame(data=data)    
df.to_csv(output_path+'/minchi2_specfittprod.csv')
del df, data

bayesian_data = {'fluxobjid':fluxobjid,
                 'logZ': np.log10(bayesian_params[:,0]),
                 'logZ_err': bayesian_errs[:,0],
                 'logtau': np.log10(bayesian_params[:,1]),
                 'logtau_err': bayesian_errs[:,1],
                 'logAv': np.log10(bayesian_params[:,2]),
                 'logAv_err': bayesian_errs[:,2],
                 'logm':bayes_mass[:,0],
                 'logm_err':bayes_mass[:,1],
                 'model_evidence':model_evidence[:]
                 }
df = pd.DataFrame(data=bayesian_data)
df.to_csv(output_path+'/bayesian_specfittprod.csv')
del df, bayesian_data

twosol_data = { 'fluxobjid':fluxobjid,
                'ev0':twosols_evidence[:,0],
                'ev1':twosols_evidence[:,1],
                'logZ0': twosols_params[:, 0],
                'logZ_err0': twosols_errs[:, 0],
                'logtau0': twosols_params[:, 1],
                'logtau_err0': twosols_errs[:, 1],
                'logAv0': twosols_params[:, 2],
                'logAv_err0': twosols_errs[:, 2],
                'logZ1': twosols_params[:, 3],
                'logZ_err1': twosols_errs[:, 3],
                'logtau1': twosols_params[:, 4],
                'logtau_err1': twosols_errs[:, 4],
                'logAv1': twosols_params[:, 5],
                'logAv_err1': twosols_errs[:, 5]
                }
df =pd.DataFrame(data=twosol_data)
df.to_csv(output_path+'/twosolprod.csv')




print('\nProcess finished!\n')





# The end.
