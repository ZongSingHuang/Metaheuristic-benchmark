# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:57:41 2021

@author: zongsing.huang
"""

import numpy as np

def aaa(X):
    # X in [-2PI 2PI], D fixed 2
    # F* = -106.7645367198034
    # X* = [4.701055751981055, 3.152946019601391], [-1.582142172055011, -3.130246799635430]
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = (X1-X2)**2
    f2 = np.sin(X1)*np.exp( (1-np.cos(X2))**2 )
    f3 = np.cos(X2)*np.exp( (1-np.sin(X1))**2 )
    
    F = f1 + f2 + f3
    
    return F

X = np.zeros([5, 2]) + [-1.582142172055011, -3.130246799635430]
F = aaa(X)