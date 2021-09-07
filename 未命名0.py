# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Camel(X):
    # [1]
    # X in [-2,-2], D fixed 2
    # X* = [-1.5, 0], [1.5, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -(-X1**4+4.5*X1**2+2)/np.exp(2*X2**2)
    
    return F

X = np.zeros([5, 2]) + [1.5, 0]
F = Camel(X)