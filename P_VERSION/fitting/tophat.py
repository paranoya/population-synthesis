#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:04:02 2019

@author: pablo
"""

import numpy as np


def tophat(arr, edges):
    tophat = np.ones_like(arr)
    tophat[arr<edges[0]]=0
    tophat[arr>edges[1]]=0
    return tophat