# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Osborne_N1(X, m=33):
    # [1]
    # X3 in [-3, 0], X1X2X4X5 in [0, 3], D fixed 5
    # X* = [0.3753, 1.9358, -1.4647, 0.01287, 0.02212]
    # F* = 5.46e-5
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    X5 = X[:, 4]
    
    j = np.arange(m) + 1
    t = 10*(j-1)
    y = np.array([0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751,
                  0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490,
                  0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406])
    
    for i in range(P):
        F[i] = np.sum( ( ( X1[i] + X2[i]*np.exp(-X4[i]*t) + X3[i]*np.exp(-X5[i]*t) ) - y )**2 )
    
    return F

X = np.zeros([5, 5]) + [0.3753, 1.9358, -1.4647, 0.01287, 0.02212]
F = Osborne_N1(X)