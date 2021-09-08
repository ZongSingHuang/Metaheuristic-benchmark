# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Mishra_N9(X):
    # [1]
    # X in [-10, 10], D fixed 3
    # X* = [1, 2, 3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    f1 = 2*X1**3 + 5*X1*X2 + 4*X3 - 2*X1**2*X3 - 18
    f2 = X1 + X2**3 + X1*X2**2 + X1*X3**2 - 22
    f3 = 8*X1**2 + 2*X2*X3 + 2*X2**2 + 3*X2**3 - 52
    
    F = ( f1*f2**2*f3 + f1*f2*f3**2 + f2**2 + (X1+X2-X3)**2 )**2
            
    return F

X = np.zeros([5, 3]) + [1, 2, 3]
F = Mishra_N9(X)