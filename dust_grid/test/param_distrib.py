#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:54:22 2019

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('bayesian_specfittprod.csv')

Z = df['met_Z']
Av = df['ext_Av']
taus = df['tau']

plt.figure()
plt.subplot(131)
plt.hist(Z[:1000])
plt.subplot(132)
plt.hist(Av[:1000])
plt.subplot(133)
plt.hist(np.log10(taus[:1000]))

plt.figure()
plt.loglog(Av[:1000], Z[:1000],'o')
plt.figure()
plt.loglog(Av[:1000], taus[:1000], 'o')