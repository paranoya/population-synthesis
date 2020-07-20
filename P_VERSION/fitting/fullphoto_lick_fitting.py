#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:08:34 2019

@author: pablo
"""

import numpy as np

import sys
sys.path.append("..")
sys.path.append("../products_codes")
import os

from glob import glob

from scipy.interpolate import interp1d, RegularGridInterpolator


import pandas as pd
from matplotlib import pyplot as plt

# =============================================================================
# How to read spectra & sample photometry
# =============================================================================

## The photometry table is ordered in terms of plate, mjd and fiber ID in the
## same way as is ordenered within the variable containing the spectrum paths


df = pd.read_csv('../data/XMATCH/crossmatch_photometry_full.csv')
sdssobjid = df['sdssID']
photometry = np.array([df['FUV_abs'], 
                       df['NUV_abs'],
                       df['u_abs'],
                       df['g_abs'], 
                       df['r_abs'],
                       df['i_abs'],
                       df['z_abs'],
                       df['J_abs'],
                       df['H_abs'],
                       df['K_abs']])
sigma_photo = np.array([df['FUV_sigma'],
                        df['NUV_sigma'],
                        df['u_sigma'],
                        df['g_sigma'], 
                        df['r_sigma'], 
                        df['i_sigma'],
                        df['z_sigma'],
                        df['J_sigma'],
                        df['H_sigma'],
                        df['K_sigma']])
labels = ['FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', 'J', 'H', 'Ks']

df = pd.read_csv('../data/XMATCH/spectral_fitting_feat.csv')
spec_features = np.array([df['d4000'],                       
                       df['MgFe']])

sigma_spec_feat = np.array([df['d4000_err'],                       
                       df['MgFe_err']])

del df
# =============================================================================
# Outputs
# =============================================================================

output_path = '../results/crossmatch'
likelihoods_path = output_path+'/likelihoods'
photo_path = output_path+'/photometry_fits'

if os.path.isdir(output_path)==False:
    os.mkdir(output_path)
    os.mkdir(likelihoods_path)
    os.mkdir(photo_path)    
    print('\nNew folders created at', output_path)

plot_fit = False
    
# =============================================================================
# Model parameters
# =============================================================================

PHOTOfolder = '../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr/'+\
                'products/photometry*'
PHOTOpaths = np.sort(glob(PHOTOfolder))

spec_feat_folder = '../population_synthesis/tau_delayedEXPSFR_BC/epoch13.648Gyr/'+\
                'products/lick_indices_Z*'
spec_feat_paths = np.sort(glob(spec_feat_folder))

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
logtau = np.log10(taus)
n_dust_models = 30
dlogtau = np.diff(np.log10(taus))[2]

# =============================================================================
# Loading models
# =============================================================================


PHOTO_models = np.empty((len(photometry), len(metallicities), len(taus),
                         n_dust_models), 
                        dtype=np.float32)

spec_models = np.empty((len(spec_features), len(metallicities), len(taus),
                         n_dust_models), 
                        dtype=np.float32)

for i_elem in range(len(metallicities)):
    
    mag = np.loadtxt(PHOTOpaths[i_elem], unpack=True)            
    Av  =np.unique(mag[1, :])
    mag = mag[2:, : ]
    PHOTO_models[:, i_elem, :, :] = mag.reshape(len(photometry),
                                                len(taus),
                                                n_dust_models)
    m_mgb, m_fe5270, m_fe5335, m_d4000 = np.loadtxt(spec_feat_paths[i_elem],
                                            usecols=(12,13,14, -1),
                                            unpack=True)            
    m_MgFe = np.sqrt(m_mgb*(0.72*m_fe5270+0.28*m_fe5335))
    spec_models[:, i_elem, :, :] = np.array([m_d4000, m_MgFe]).reshape(
                            len(spec_features), len(taus), n_dust_models)
logAv = np.log10(Av) 
dlogAv = np.diff(logAv)[2]
       
del mag, i_elem


# =============================================================================
# Metalliciticy interpolation
# =============================================================================
print('\nModel Metallicity interpolation...\n')
metallicities = np.array(metallicities, dtype=float)
n_metallicities_interpolation = 30

new_metallicities = np.log10(np.logspace(
                    np.log10(metallicities[1]),
                    np.log10(metallicities[-1]),
                    n_metallicities_interpolation)
                                        )
PHOTO_models = interp1d(np.log10(metallicities), PHOTO_models, axis=1)(new_metallicities)
spec_models = interp1d(np.log10(metallicities), spec_models, axis=1)(new_metallicities)
metallicities = new_metallicities
dlogZ = np.diff(metallicities)[2]

del new_metallicities, n_metallicities_interpolation

masses = np.logspace(6.5, 13.5, 200)
logmasses = np.log10(masses)
dlogmass = np.diff(np.log10(masses))[2]
delta_mag = 2.5*np.log10(masses)

PHOTO_models = PHOTO_models[:,:,:,:, np.newaxis]-delta_mag[np.newaxis,
                                        np.newaxis, np.newaxis, np.newaxis, :]
PHOTO_models = np.float32(PHOTO_models)
del delta_mag

logZ = metallicities

print('\n Model grid ready with dimensions '+format(PHOTO_models.shape))
###############################################################################
#########################    Spectrum fitting    ##############################
###############################################################################
#%%


bayesian_params = np.zeros((photometry.shape[1], 3))
bayesian_errs = np.zeros((photometry.shape[1], 3))

model_evidence = np.zeros((photometry.shape[1]))

bayes_mass = np.zeros((photometry.shape[1], 2), dtype=np.float16)

print('\nStarting photometric fit...\n')
dp_dmdtau_all = 0
#for galaxy_i in range(photometry.shape[1]):
for galaxy_i in range(20000):
# for galaxy_i in [1]:
    print('----------------------------')
    print('galaxy #{}'.format(galaxy_i))
    ## ---- PHOTOMETRY / MASS INFERENCE ---
    gal_photo = photometry[:, galaxy_i]
    gal_spec_feat = spec_features[:, galaxy_i]
    gal_sigma_photo = sigma_photo[:, galaxy_i]
    gal_sigma_spec = sigma_spec_feat[:, galaxy_i]
    
    photo_chi2 = (PHOTO_models -\
            gal_photo[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])**2 \
         /gal_sigma_photo[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]**2
    spec_chi2 =  (spec_models-gal_spec_feat[:, np.newaxis, np.newaxis, np.newaxis])**2 \
         /gal_sigma_spec[:, np.newaxis, np.newaxis, np.newaxis]**2
    
    photo_chi2 = np.mean(photo_chi2, axis=0) #/np.mean(photo_chi2)
    spec_chi2 = np.mean(spec_chi2, axis=0) #/np.mean(photo_chi2)
    lhood = np.exp(-photo_chi2/2-spec_chi2[:,:,:, np.newaxis]/2)
    # lhood = np.exp(-photo_chi2/2)

    norm = np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)
    
    dpdlogZ  = np.sum(lhood*dlogtau*dlogAv*dlogmass, axis=(1,2,3))\
                            /norm
    
    dpdlogtau  = np.sum(lhood*dlogZ*dlogAv*dlogmass, axis=(0,2,3))\
                            /norm
    
    dpdlogAv  = np.sum(lhood*dlogZ*dlogtau*dlogmass, axis=(0,1,3))\
                            /norm
    
    dpdlogm  = np.sum(lhood*dlogZ*dlogAv*dlogtau, axis=(0,1,2))\
                            /norm
    
                            
    gal_logZ = np.sum(dpdlogZ*logZ*dlogZ)
    gal_logtau = np.sum(dpdlogtau*logtau*dlogtau)
    gal_logAv = np.sum(dpdlogAv*logAv*dlogAv)
    gal_logm = np.sum(dpdlogm*logmasses*dlogmass)
    
    sigma_logZ = np.sum(dpdlogZ*(logZ-gal_logZ)**2*dlogZ)
    sigma_logtau = np.sum(dpdlogtau*(logtau-gal_logtau)**2*dlogtau)
    sigma_logAv = np.sum(dpdlogAv*(logAv-gal_logAv)**2*dlogAv)
    sigma_logm = np.sum(dpdlogm*(logmasses-gal_logm)**2*dlogmass)                             
    
    
    bayesian_params[galaxy_i, :] = [gal_logZ, gal_logtau, gal_logAv]
    bayesian_errs[galaxy_i, :] = [sigma_logZ, sigma_logtau, sigma_logAv]
    
    bayes_mass[galaxy_i, :]  = [gal_logm, sigma_logm]
    
    if plot_fit==True:
        minchi2_pos = np.where(photo_chi2==np.min(photo_chi2))

        dpdlogZdlogtau  = np.sum(lhood*dlogAv*dlogmass, axis=(2,3))\
                                /np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)
        dpdlogZdlogAv  = np.sum(lhood*dlogtau*dlogmass, axis=(1,3))\
                                /np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)                            
        dpdlogtaudlogAv  = np.sum(lhood*dlogZ*dlogmass, axis=(0,3))\
                                /np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)
        dpdlogtaudlogm  = np.sum(lhood*dlogAv*dlogZ, axis=(0,2))\
                                /np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)
        dpdlogAvdlogm  = np.sum(lhood*dlogtau*dlogZ, axis=(0,1))\
                                /np.sum(lhood*dlogZ*dlogAv*dlogtau*dlogmass)                            
    
    plogzlogtaulogav = 1/(
            metallicities[-1]-metallicities[0]+\
            np.log10(taus[-1]/taus[0])+np.log10(Av[-1]/Av[0]))
    
    evidence = np.sum(plogzlogtaulogav*lhood*dlogZ*dlogtau*dlogAv)#\
                                            #/np.sum(lhood*dlogZ*dlogtau*dlogAv)
    model_evidence[galaxy_i] = evidence
        
    
    print('Model evidence:', model_evidence[galaxy_i])
    if plot_fit==True:    
        plt.figure(figsize=(13,13))
        plt.subplot(331)
        plt.contourf(logZ, logtau, dpdlogZdlogtau.T , origin='lower',
                     cmap='magma',
                     levels=40
                     )
        plt.ylabel(r'$\log(\tau)$', fontsize=14)
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False)
        plt.subplot(332)
        plt.contourf(logmasses, logtau, dpdlogtaudlogm , origin='lower',
                     cmap='magma',
                     levels=40
                     )
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False, labelleft=False)
        plt.subplot(333)
        plt.plot(dpdlogtau, logtau, '-o', color='k')
        plt.axhspan(gal_logtau-sigma_logtau, gal_logtau+sigma_logtau, 
                    color='cyan', alpha=0.2, 
                    label=r'$\log(\tau/Gyr)={:.2} \pm {:.2}$'.format(gal_logtau,
                                 sigma_logtau))
        plt.xlabel(r'$\frac{dp}{d\log(\tau)}$', fontsize=14)
        plt.legend()
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False, labelleft=False)
        plt.subplot(334)
        plt.contourf(logZ, logAv, dpdlogZdlogAv.T , origin='lower',
                     cmap='magma',
                     levels=40
                     )
        plt.ylabel(r'$\log(A_v)$', fontsize=14)
    
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False, labelleft=False)
        plt.subplot(335)
        plt.contourf(logmasses, logAv, dpdlogAvdlogm , origin='lower',
                     cmap='magma',
                     levels=40
                     )
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False, labelleft=False)
        plt.subplot(336)
        plt.plot(dpdlogAv, logAv, '-o', color='k')
        plt.axhspan(gal_logAv-sigma_logAv, gal_logAv+sigma_logAv, 
                    color='cyan', alpha=0.2,
                    label=r'$\log(A_V/mag)={:.2} \pm {:.2}$'.format(gal_logAv,
                             sigma_logAv))
        plt.legend()
        plt.xlabel(r'$\frac{dp}{d\log(A_V)}$', fontsize=14)
    
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=False, labelleft=False)
        plt.subplot(337)
        plt.plot(logZ, dpdlogZ, '-o', color='k')
        plt.axvspan(gal_logZ-sigma_logZ, gal_logZ+sigma_logZ, 
                    color='cyan', alpha=0.2,
                    label=r'$\log(Z)={:.2} \pm {:.2}$'.format(gal_logZ,
                                 sigma_logZ))
        plt.legend()
        plt.xlabel(r'$\log(Z)$', fontsize=14)
        plt.ylabel(r'$\frac{dp}{d\log(Z)}$', fontsize=14)
    
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=True, labelleft=False)
        plt.subplot(338)
        plt.plot(logmasses, dpdlogm, '-o', color='k')
        plt.axvspan(gal_logm-sigma_logm, gal_logm+sigma_logm, 
                    color='cyan', alpha=0.2,
                    label=r'$\log(M/M_\odot)={:3.1} \pm {:.1}$'.format(gal_logm, 
                                 sigma_logm))
        plt.xlabel(r'$\log(M/M_\odot)$', fontsize=14)
        plt.ylabel(r'$\frac{dp}{d\log(M)}$', fontsize=14)
        plt.legend()
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=True, labelleft=False)
        plt.subplot(339)
        plt.errorbar(labels, gal_photo, yerr=gal_sigma_photo, fmt='o', 
                     color='k', label='Observed')
        plt.plot(PHOTO_models[:, minchi2_pos[0], minchi2_pos[1], 
                              minchi2_pos[2], minchi2_pos[3]], 'r-^',
                        label=r'Min $\chi^2$={:.4}'.format(np.min(photo_chi2)))
        plt.ylim(gal_photo[gal_photo<0][-1]-0.5, gal_photo[gal_photo<0][0]+0.5)
    #    plt.tick_params(axis='both', direction='in',
    #                    bottom=True, top=True, left=True, right=True, 
    #                    labelbottom=True, labelleft=False)
        plt.legend()                      
        plt.subplots_adjust(wspace=0.35, hspace=0.35)
        plt.savefig(photo_path+'/photospec_fit_'+str(galaxy_i)+'.png')
        plt.close()
        
        
    
    
    
# =============================================================================
#  OUTPUT DATA        
# =============================================================================
##print('No saving data')
print('\nSaving data...\n')

bayesian_data = {'sdssobjid':sdssobjid,
                 'logZ': bayesian_params[:,0],
                 'logZ_err': bayesian_errs[:,0],
                 'logtau': bayesian_params[:,1],
                 'logtau_err': bayesian_errs[:,1],
                 'logAv': bayesian_params[:,2],
                 'logAv_err': bayesian_errs[:,2],
                 'logm':bayes_mass[:,0],
                 'logm_err':bayes_mass[:,1],
                 'model_evidence':model_evidence[:]
                 }
df = pd.DataFrame(data=bayesian_data)
df.to_csv(output_path+'/bayesian_specfittprod.csv')
del df, bayesian_data




print('Output data can be found at: \n'+ output_path)
print('\nProcess finished!\n')





# The end.
