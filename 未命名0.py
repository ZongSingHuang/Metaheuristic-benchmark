# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Simpleton(X):
    # [1]
    # X in [1, 10], D fixed 10
    # X* = [10, 10, 10, 10, 10, 1, 1, 1, 1, 1]
    # F* = 1E5
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    X5 = X[:, 4]
    X6 = X[:, 5]
    X7 = X[:, 6]
    X8 = X[:, 7]
    X9 = X[:, 8]
    X10 = X[:, 9]
    
    F = X1*X2*X3*X4*X5/(X6*X7*X8*X9*X10)
    
    return F

X = np.zeros([5, 10]) + [10, 10, 10, 10, 10, 1, 1, 1, 1, 1]
F = Simpleton(X)