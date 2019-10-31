#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:04:05 2019

@author: pablo
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


gama_bayesian = pd.read_csv('bayesian_assignment_GAMA.csv')


Mass = gama_bayesian['Mass']
tau = gama_bayesian['Tau']
E =  gama_bayesian['E(B-V)']
Av = E*3.1
met = gama_bayesian['Met_Z']



plt.figure()
plt.subplot(121)
plt.plot(np.log10(Mass), Av, ',')
plt.subplot(122)
plt.semilogx(tau, Av, ',')

plt.figure()
plt.plot(np.log10(Mass), np.log10(tau), ',')

H = np.histogram2d(np.log10(Mass), np.log10(tau), 
                   bins=20, range=[[8,11.5],[-0.7, 1.7]])

plt.figure()
plt.contourf(H[1][:-1], H[2][:-1], H[0].T)