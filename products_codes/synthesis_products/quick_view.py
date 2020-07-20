#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:26:28 2020

@author: pablo
"""

import numpy as np 
from matplotlib import pyplot as plt
from glob import glob



photo_path = '../../population_synthesis/EXP_delayed_tau_BC/epoch13.7Gyr/'+\
                                                'products/photometry.txt'
lick_path = '../../population_synthesis/EXP_delayed_tau_BC/epoch13.7Gyr/'+\
                                                'products/lick_indices.txt'

photo = np.loadtxt(photo_path)

u = photo[:,4].reshape(30, 80)
g = photo[:,5].reshape(30, 80)
r = photo[:,6].reshape(30, 80)

plt.imshow(u-r, aspect='auto')
plt.colorbar()

