# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Hosaki(X):
    # [1]
    # X in [0, 10], D fixed 2
    # X* = [4, 2]
    # F* = -2.345811576101292
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1-8*X1+7*X1**2-7/3*X1**3+0.25*X1**4) * X2**2*np.exp(-X2)
    
    return F

X = np.zeros([5, 2]) + [4, 2]
F = Hosaki(X)