# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def CosineMixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0.1*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 0.1*np.sum( np.cos(5*np.pi*X), axis=1 ) - np.sum(X**2, axis=1)
    
    return F

X = np.zeros([5, 7])
F = CosineMixture(X)