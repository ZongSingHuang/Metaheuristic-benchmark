# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:08:21 2021

@author: zongsing.huang
"""

import numpy as np

def Damavandi(X):
    # [1]
    # X in [0, 14], D fixed 2
    # X* = [2, 2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    mask1 = X1==2
    mask2 = X2==2
    mask3 = mask1 * mask2
    mask4 = ~mask3
    f1 = 1 - np.abs( np.sin(np.pi*(X1[mask4]-2))*np.sin(np.pi*(X2[mask4]-2))/(np.pi**2*(X1[mask4]-2)*(X2[mask4]-2)) )**5
    f2 = 2 + (X1[mask4]-7)**2 + 2*(X2[mask4]-7)**2
    F[mask4] = f1 * f2
    
    return F

X = np.zeros([5, 2]) + 2
F = Damavandi(X)#[2.00000000000001, 1.999999999999999]