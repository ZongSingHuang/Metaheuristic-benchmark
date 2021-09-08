# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Peaks(X):
    # [1]
    # X in [-4, 4], D fixed 2
    # X* = [0.228279999979237, -1.625531071954464]
    # F* = -6.551133332622496
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = 3*(1-X1)**2 * np.exp(-X1**2-(X2+1)**2)
    f2 = 10*(X1/5-X1**3-X2**5) * np.exp(-X1**2-X2**2)
    f3 = np.exp(-(X1+1)**2-X2**2) / 3
    F = f1 - f2 - f3
    
    return F



X = np.zeros([5, 2]) + [0.228279999979237, -1.625531071954464]
F = Peaks(X)