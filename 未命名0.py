# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def SchmidtVetter(X):
    # [1]
    # X in [0, 10], D fixed 3
    # X* = [0.78547, 0.78547, 0.78547]
    # F* = 3
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    f1 = 1 + (X1-X2)**2
    f2 = np.sin( (np.pi*X2+X3)/2 )
    f3 = np.exp( ((X1+X2)/X2 - 2)**2 )
    
    F = 1/f1 + f2 + f3
            
    return F

X = np.zeros([5, 3]) + [0.78547, 0.78547, 0.78547]
F = SchmidtVetter(X)