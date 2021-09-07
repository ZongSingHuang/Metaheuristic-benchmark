# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Zilinskas_N3(X):
    # [1]
    # X in [0, 100], D fixed 1
    # X* = 3PI/2 + 2PI = [10.995574287564276]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = np.sin(X1)
    
    return F

X = np.zeros([5, 1]) + 10.995574287564276
F = Zilinskas_N3(X)