# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Six_Hump_Camel_Back(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [-0.08984201368301331, 0.7126564032704135] or [0.08984201368301331, -0.7126564032704135]
    # F* = -1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + X1**6/3 + X1*X2 - 4*X2**2 + 4*X2**4
    
    return F



X = np.zeros([5, 2]) + [-0.08984201368301331, -0.7126564032704135]
F = Six_Hump_Camel_Back(X)