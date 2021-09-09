# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

# =============================================================================
# main ref
# [1] https://www.al-roomi.org/benchmarks
# =============================================================================

import numpy as np

#%%
# =============================================================================
# 4-D
# =============================================================================
def CoranaParabola(X):
    # [1]
    # X in [-100, 100], D fixed 4
    # X* = [0, 0, 0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    F = np.zeros([P])
    s = 0.2
    d = np.array([1, 1000, 10, 100])
    z = 0.2*np.floor( np.abs(X/s)+0.49999 ) * np.sign(X)
    
    for i in range(P):
        for j in range(D):
            if np.abs(X[i, j]-z[i, j])<0.05:
                F[i] = F[i] + 0.15 * d[j] * ( z[i, j]-0.05*np.sign(z[i, j]) )**2
            else:
                F[i] = F[i] + d[j]*X[i]**2
    return F

def Gear(X):
    # [1]
    # X in [12, 60], D fixed 4
    # X* = [16, 19, 43, 49], 其中X1可以和X2對調；X3可以和X4對調，例如[19, 16, 49, 43]或者[19, 16, 43, 49]
    # F* = 2.700857148886513e-12
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    F = ( 1/6.931 - (np.floor(X1)*np.floor(X2))/(np.floor(X3)*np.floor(X4)) )**2
    
    return F

def MieleCantrell(X):
    # [1]
    # X in [-1, 1], D fixed 4
    # X* = [0, 1, 1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    F = (np.exp(X1)-X2)**4 + 100*(X2-X3)**6 + np.tan(X3-X4)**4 + X1**8
    
    return F

def PowellQuartic(X):
    # Powell’s Singular Function
    # [1]
    # X in [-10, 10], D fixed 4
    # X* = [0, 0, 0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    F = (X1+10*X2)**2 + 5*(X3-X4)**2 + (X2-2*X3)**4 + 10*(X1-X4)**4
    
    return F

def Wood(X):
    # [1]
    # Colville's Function
    # X in [-10, 10], D fixed 4
    # X* = [1, 1, 1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    F = (100*(X2-X1**2))**2 + (1-X1)**2 + 90*(X4-X3**2)**2 + (1-X3)**2 + 10.1*((X2-1)**2+(X4-1)**2) + 19.8*(X2-1)*(X4-1)
    
    return F