# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Zettl(X):
    # X in [-1, 5], D fixed 2
    # X* = [-0.02989597760285287, 0]
    # F* = -0.003791237220468656
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.25*X1 + ( X1**2 - 2*X1 + X2**2 )**2
    
    return F



X = np.zeros([5, 2]) + [-0.02989597760285287, 0]
F = Zettl(X)