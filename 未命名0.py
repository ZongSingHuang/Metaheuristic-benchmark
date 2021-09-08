# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Tsoulos(X):
    # [1]
    # X in [-1, 1], D fixed 2
    # X* = [0, 0]
    # F* = -2
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + X2**2 - np.cos(18*X1) - np.cos(18*X2)
    
    return F



X = np.zeros([5, 2]) + [0, 0]
F = Tsoulos(X)