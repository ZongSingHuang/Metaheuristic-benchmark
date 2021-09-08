# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

# =============================================================================
# main ref
# [1] https://www.al-roomi.org/benchmarks
# [2] A Fuzzy Adaptive Simplex Search Optimization Algorithm
# [3] http://infinity77.net/global_optimization/genindex.html
# [4] https://www.sfu.ca/~ssurjano/optimization.html
# =============================================================================

import numpy as np

#%%
# =============================================================================
# 3-D
# =============================================================================
def BoxBettExponentialQuadraticSum(X):
    # [1]
    # X1 in [0.9, 1.2], X2 in [9, 11.2], X3 in [0.9, 1.2], D fixed 3
    # X* = [1, 10, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    L = np.arange(10) + 1
    
    for i in range(P):
        F[i] = np.sum( np.exp(-0.1*L*X1[i]) - np.exp(-0.1*L*X2[i]) - (np.exp(-0.1*L)-np.exp(-L))*X3[i] )
    
    return F

def FletcherPowellHelicalValley(X):
    # [1], [2]
    # X in [-100, 100], D fixed 3
    # X* = [1, 0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    Theta = [ np.arctan(X2[i]/X1[i]) if X1[i]>=0 else np.pi+np.arctan(X2[i]/X1[i]) for i in range(P) ]
    Theta = np.array(Theta)/(2*np.pi)
    
    F = 100 * ( (X3-10*Theta**2) + ((X1**2+X2**2)**0.5-1)**2 ) +X3**2
    
    return F

def GulfResearch(X, m=99):
    # [1], [3]
    # X1 in [0.1, 100], X2 in [0, 25.6], X3 in [0, 5], D fixed 3
    # X* = [50, 25, 1.5]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    
    tj = (np.arange(m)+1)/100
    yj = 25 + ( -50*np.log(tj) )**(2/3)
    
    for i in range(P):
        F[i] = np.sum( np.exp( -np.abs(yj-X2[i])**X3[i]/X1[i] ) - tj )
    
    return F

def Hartmann_N1(X):
    # [1], [3]
    # X in [1, 3], D fixed 3
    # X* = [0.114614, 0.555649, 0.852547]
    # F* = -3.86278
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    a = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35], 
                  [3.0, 10, 30], 
                  [0.1, 10, 35]])
    p = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])
    
    for k in range(P):
        first = a[0] * np.exp(-1*np.sum(A[0]*(X[k]-p[0])**2))
        second = a[1] * np.exp(-1*np.sum(A[1]*(X[k]-p[1])**2))
        third = a[2] * np.exp(-1*np.sum(A[2]*(X[k]-p[2])**2))
        fourth = a[3] * np.exp(-1*np.sum(A[3]*(X[k]-p[3])**2))
        
        F[k] = -1*(first + second + third + fourth)
            
    return F

def Holzman_N1(X):
    # 解不出來!!!
    # [1]
    # X1 in [0.1, 100], X2 in [0, 25.6], X3 in [0, 5], D fixed 3
    # X* = [50, 25, 1.5]
    # F* = -3.86278
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    F = np.zeros([P])
    L = np.arange(100)
    ui = 25 + ( -50*np.log(0.01*(L+1)) )**(2/3)
    
    for i in range(P):
        F[i] = np.sum( -0.1*(L+1) + np.exp((ui-X2[i])**X3[i]/X1[i]) )
            
    return F

def MeyerRoth(X):
    # [1]
    # X in [0, 1], D fixed 3
    # X* = [3.13, 15.16, 0.78]
    # F* = 0.00001
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    tj = np.array([1, 2, 1, 2, 0.1])
    vj = np.array([1, 1, 2, 2, 0])
    yj = np.array([0.126, 0.219, 0.076, 0.126, 0.186])
    
    for i in range(P):
        F[i] = np.sum( ( X1[i]*X3[i]*tj/(1+X1[i]*tj+X2[i]*vj) - yj )**2 )
            
    return F

def Mishra_N9(X):
    # Dodecal Polynomial Function
    # [1]
    # X in [-10, 10], D fixed 3
    # X* = [1, 2, 3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    f1 = 2*X1**3 + 5*X1*X2 + 4*X3 - 2*X1**2*X3 - 18
    f2 = X1 + X2**3 + X1*X2**2 + X1*X3**2 - 22
    f3 = 8*X1**2 + 2*X2*X3 + 2*X2**2 + 3*X2**3 - 52
    
    F = ( f1*f2**2*f3 + f1*f2*f3**2 + f2**2 + (X1+X2-X3)**2 )**2
            
    return F

def SchmidtVetter(X):
    # [1]
    # X in [0, 10], D fixed 3
    # X* = [0.78547, 0.78547, 0.78547]
    # F* = 3
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    f1 = 1 + (X1-X2)**2
    f2 = np.sin( (np.pi*X2+X3)/2 )
    f3 = np.exp( ((X1+X2)/X2 - 2)**2 )
    
    F = 1/f1 + f2 + f3
            
    return F

def Wolfe(X):
    # [1]
    # X in [0, 2], D fixed 3
    # X* = [0, 0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    
    F = 4/3 * (X1**2 + X2**2 - X1*X2)**0.75 + X3
    
    return F