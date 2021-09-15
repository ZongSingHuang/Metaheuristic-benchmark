# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:57:41 2021

@author: zongsing.huang
"""

import numpy as np

def aaa(X):
    # X in [-50, 50]
    # F* = 0
    # X* = [1, 1, ..., 1]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    X1 = X[:, 0]
    XD = X[:, -1]
    Xi = X[:, :-1]
    Xi_1 = X[:, 1:]
    
    f1 = np.sin(3*np.pi*X1)**2
    f2 = np.sum( (Xi-1)**2 * (1+np.sin(3*np.pi*Xi_1)**2), axis=1 )
    f3 = (XD-1)**2 * (1+np.sin(2*np.pi*XD)**2)
    
    F = 0.1 * (f1+f2+f3) + u(X, 5, 100, 4)
    
    return F

#%%
def y(X):
    F = 1 + (X+1)/4
    return F

def u(X, a, k, m):
    F = np.zeros_like(X)
    mask1 = X>a
    mask3 = X<-a
    mask2 = ~(mask1+mask3)

    F[mask1] = k*(X[mask1]-a)**m
    F[mask2] = 0
    F[mask3] = k*(-X[mask3]-a)**m
    
    F = np.sum(F, axis=1)
    
    return F

X = np.zeros([5, 4]) + 1
F = aaa(X)