#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:01:17 2019

@author: pablo
"""
import numpy as np

class expectation_maximization(object):
    
    def __init__(self, **kwargs):
        self.prob_distrib = kwargs['prob_distrib']
        self.x = kwargs['x'] # 0'th dim
        self.y = kwargs['y'] # 1st dim
        self.z = kwargs['z'] # 2nd dim
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, 
                                             indexing='ij')        
        self.mu0 = np.array([self.x[0], self.y[0], self.z[0]])
        self.mu1 = np.array([self.x[-1], self.y[-1], self.z[-1]])
        self.sigma0 = np.array([self.x[-1]-self.x[0], 
                                self.y[-1]-self.y[0], 
                               self.z[-1]-self.z[0]])/3
        self.sigma1 = self.sigma0
        
        expectation_maximization.iterative_maximization(self)
        
        while ((np.abs(self.old_mu0-self.mu0)/self.mu0).any()>0.05)&(
                (np.abs(self.old_mu1-self.mu1)/self.mu1).any()>0.05):
            expectation_maximization.iterative_maximization(self)
        
        self.sigma0 = np.array([self.new_sigmaX0,
                                self.new_sigmaY0,
                                self.new_sigmaZ0])
        self.sigma1 = np.array([self.new_sigmaX1,
                                self.new_sigmaY1,
                                self.new_sigmaZ1])
    
    def gauss_distrib(self, mu, sigma):
        
        
        self.R2 = (((self.X-mu[0])/sigma[0])**2 + ((self.Y-mu[1])/sigma[1])**2 +\
              ((self.Z-mu[2])/sigma[2])**2 )
             
#        gaussian = np.exp(-R2/2)
#        return gaussian
        return 1/(1+self.R2)
    
    def iterative_maximization(self):
        
        new_mu0 = self.mu0
        new_sigma0 = self.sigma0
        new_mu1 = self.mu1
        new_sigma1 = self.sigma1
                
        w0 = self.gauss_distrib(new_mu0, new_sigma0)
        w1 = self.gauss_distrib(new_mu1, new_sigma1)
        region0 = np.where(w0 >= w1)
        region1 = np.where(w1 >= w0)
        
        prob0 = self.prob_distrib[region0]
        self.ev0 = np.nanmean(prob0, dtype=np.float16)
#        print('Ev0: ', self.ev0)
        prob0 /= np.nansum(prob0)
        new_X0 = np.nansum(prob0*self.X[region0])
        new_Y0 = np.nansum(prob0*self.Y[region0])
        new_Z0 = np.nansum(prob0*self.Z[region0])
        
        self.new_sigmaX0 = np.nansum(prob0*(new_X0-self.X[region0])**2)
        self.new_sigmaY0 = np.nansum(prob0*(new_Y0-self.Y[region0])**2)
        self.new_sigmaZ0 = np.nansum(prob0*(new_Z0-self.Z[region0])**2)

        prob1 = self.prob_distrib[region1]
        self.ev1 = np.nanmean(prob1, dtype=np.float16)
#        print('Ev1: ', self.ev1)
        prob1 /= np.nansum(prob1)
        new_X1 = np.nansum(prob1*self.X[region1])
        new_Y1 = np.nansum(prob1*self.Y[region1])
        new_Z1 = np.nansum(prob1*self.Z[region1])
        
        self.new_sigmaX1 = np.nansum(prob1*(new_X1-self.X[region1])**2)
        self.new_sigmaY1 = np.nansum(prob1*(new_Y1-self.Y[region1])**2)
        self.new_sigmaZ1 = np.nansum(prob1*(new_Z1-self.Z[region1])**2)

#        print(new_X0, new_Y0, new_Z0)
#        print(new_X1, new_Y1, new_Z1)
        
        self.old_mu0 = self.mu0
        self.old_mu1 = self.mu1
        
        self.mu0 = np.array([new_X0, new_Y0, new_Z0])
        self.mu1 = np.array([new_X1, new_Y1, new_Z1])
#        self.sigma0 = np.array([new_sigmaX0, new_sigmaY0, new_sigmaZ0])
#        self.sigma1 = np.array([new_sigmaX1, new_sigmaY1, new_sigmaZ1])

#    def old_iterative_maximization(self):
#        R0 = w0*self.prob_distrib/(w0+w1)
#        R1 = w1*self.prob_distrib/(w0+w1)
#        
#        new_mu0 = np.nansum(R0[np.newaxis, :, : , :]*[self.X, self.Y, self.Z], 
#                         axis=(1,2,3))/np.sum(R0)            
#        var = [self.X, self.Y, self.Z]-new_mu0[:, np.newaxis,
#                                                  np.newaxis, np.newaxis]
#        
#        new_sigma0 = np.nansum(R0[np.newaxis, :, : , :]*
#                            var**2, axis=(1,2,3))/np.nansum(R0)
#        
#        new_mu1 = np.nansum(R1[np.newaxis, :, : , :]*[self.X, self.Y, self.Z], 
#                         axis=(1,2,3))/np.sum(R1)            
#        var = [self.X, self.Y, self.Z]-new_mu1[:, np.newaxis, 
#                                                  np.newaxis, np.newaxis]
#        new_sigma1 = np.nansum(R1[np.newaxis, :, : , :]*
#                            var**2, axis=(1,2,3))/np.nansum(R1)
#        
#        self.mu0 = new_mu0
#        self.sigma0 = new_sigma0    
#        self.mu1 = new_mu1
#        self.sigma1 = new_sigma1
#        print(new_mu0, new_sigma0)
#        print(new_mu1, new_sigma1)


# Fin
            
            
            