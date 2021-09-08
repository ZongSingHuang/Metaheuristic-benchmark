# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Price_N4(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0, 0], [2, 4], [1.464352119663698, -2.506012760781662]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (2*X1**3*X2-X2**3)**2 + (6*X1-X2**2+X2)**2
    
    return F



X = np.zeros([5, 2]) + [1.464352119663698, -2.506012760781662]
F = Price_N4(X)