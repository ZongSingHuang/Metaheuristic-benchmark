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
# 9-D
# =============================================================================
def ANNsXOR(X):
    # [1]
    # X in [-1, 1], D fixed 9
    # X* = [0.99999, 0.99993, -0.89414, 0.99994, 0.55932, 0.99994, 0.99994, -0.99963, -0.08272]
    # F* = 0.959759
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
    
    f1 = -X7/(1+np.exp(-(X1+X2+X5))) - X8/(1+np.exp(-(X3+X4+X6))) - X9
    f2 = -X7/(1+np.exp(-X5)) - X8/(1+np.exp(-X6)) - X9
    f3 = -X7/(1+np.exp(-(X1+X5))) - X8/(1+np.exp(-(X3+X6))) - X9
    f4 = -X7/(1+np.exp(-(X2+X5))) - X8/(1+np.exp(-(X4+X6))) - X9
    
    f1 = ( 1 + np.exp(f1) )**-2
    f2 = ( 1 + np.exp(f2) )**-2
    f3 = ( 1 - 1/(1+np.exp(f3)) )**2
    f4 = ( 1 - 1/(1+np.exp(f4)) )**2
    
    F = f1 + f2 + f3 + f4
    
    return F

def PriceTransistorModelling(X):
    # [1]
    # X in [-10, 10], D fixed 9
    # X* = [0.9, 0.45, 1,2, 8, 8, 5, 1, 2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    X5 = X[:, 4]
    X6 = X[:, 5]
    X7 = X[:, 6]
    X8 = X[:, 7]
    X9 = X[:, 8]
    
    g = np.array([[0.485000, 0.752000, 0.869000, 0.982000],
                  [0.369000, 1.254000, 0.703000, 1.455000],
                  [5.209500, 10.06770, 22.92740, 20.21530],
                  [23.30370, 101.7790, 111.4610, 191.2670],
                  [28.51320, 111.8467, 134.3884, 211.4823]])
    
    r = X1*X3 - X2*X4
    for i in range(P):
        a = (1-X1[i]*X2[i]) * X3[i] * ( np.exp(X5[i]*(g[0]-g[2]*X7[i]*1e-3-g[4]*X8[i]*1e-3)) - 1 ) - g[4] + g[3]*X2[i]
        b = (1-X1[i]*X2[i]) * X4[i] * ( np.exp(X6[i]*(g[0]-g[1]-g[2]*X7[i]*1e-3+g[3]*X9[i]*1e-3)) - 1 ) - g[4]*X1[i] + g[3]
        
        F[i] = r[i]**2 + np.sum(a**2+b**2)
        
    return F