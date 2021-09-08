# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def CoranaParabola(X):
    # [1]
    # X in [-100, 100], D fixed 4
    # X* = [0, 0, 0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    si = 0.2
    di = np.tile( np.array([1, 1000, 10, 100]), P).reshape(P, 4)
    zi = 0.2*np.floor( np.abs(X/si)+0.49999 ) * np.sign(X)
    
    mask1 = np.abs(X-zi)<0.05
    mask2 = ~mask1
    F[mask1] = 0.15*di[mask1]*(zi[mask1]-0.05*np.sign(zi[mask1]))**2
    F[mask2] = di[mask2]*X[mask2]**2
    
    return F

X = np.zeros([5, 4]) + [0, 0, 0, 0]
F = CoranaParabola(X)