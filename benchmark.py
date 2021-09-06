# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

# =============================================================================
# main ref
# A new mutation operator for real coded genetic algorithms, 2007, https://doi.org/10.1016/j.amc.2007.03.046
# A Literature Survey of Benchmark Functions For Global Optimization Problems, 2013, https://arxiv.org/pdf/1308.4008.pdf
# An Efficient Real-coded Genetic Algorithm for Real- Parameter Optimization, 2010, https://doi.org/10.1109/ICNC.2010.5584209
# A novel enhanced whale optimization algorithm for global optimization, 2021, https://doi.org/10.1016/j.cie.2020.107086
# MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization, http://www.geocities.ws/eadorio/mvf.pdf
# http://infinity77.net/global_optimization/genindex.html
# http://benchmarkfcns.xyz/fcns
# https://www.al-roomi.org/benchmarks
# https://www.sfu.ca/~ssurjano/optimization.html
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm
# https://www.indusmic.com/blog/categories/benchmark-function
# https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda#ackley-function
# https://en.wikipedia.org/wiki/Test_functions_for_optimization
# https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/index.html
# =============================================================================

import numpy as np

def Ackley1(X):
    # X in [-32, 32]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    left = -20 * np.exp( -0.2 * ( 1/D * np.sum(X**2, axis=1) )**0.5 )
    right = np.exp( 1/D * np.sum( np.cos(2*np.pi*X), axis=1 ) )
    
    F = left - right + 20 + np.e
    
    return F

def Ackley2(X):
    # X in [-32, 32], D fixed 2
    # X* = [0, 0]
    # F* = -200
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -200*np.exp(-0.2*(X1**2+X2**2)**0.5)
    
    return F

def Ackley3(X):
    # X in [-32, 32], D fixed 2
    # X* = [±0.682584587365898, -0.36075325513719]
    # F* = -195.629028238419
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -200*np.exp(-0.02*(X1**2+X2**2)**0.5) + 5*np.exp(np.cos(3*X1)+np.sin(3*X2))
    
    return F

def Ackley4(X):
    # X in [-35, 35]
    # X*(D=2) = [-1.51, -0.755]
    # F*(D=2) = -4.590101633799122
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.exp(-0.2)*(X[:, :-1]**2+X[:, 1:]**2)**0.5 + 3*(np.cos(2*X[:, :-1])+np.sin(2*X[:, 1:])), axis=1 )
    
    return F

def Adjiman(X):
    # X in [(-1, 2), (-1, 1)] or X in [-5, 5]
    # X* = [x1的最大值, 0]
    # F* = x1的最大值
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.cos(X1)*np.sin(X2) - X1/(X2**2+1)
    
    return F

def Alpine1(X):
    # X in [0, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.abs( X*np.sin(X)+0.1*X ), axis=1 )
    
    return F

def Alpine2(X):
    # X in [-10, 10]
    # X* = [7.9170526982459462172, 7.9170526982459462172, ..., 7.9170526982459462172]
    # F* = 2.8081311800070053291**D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.prod( np.sin(X)*X**0.5, axis=1 )
    
    return F

def Bartels_Conn(X):
    # X in [-500, 500], D fixed 2
    # X* = [0, 0]
    # F* = 1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.abs(X1**2+X2**2+X1*X2) + np.abs(np.sin(X1)) + np.abs(np.cos(X2))
    
    return F

def Beale(X):
    # X in [-4.5, 4.5], D fixed 2
    # X* = [3, 0.5]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1.5-X1+X1*X2)**2 + (2.25-X1+X1*X2**2)**2 + (2.625-X1+X1*X2**3)**2
    
    return F

def Bird(X):
    # X in [-2pi, 2pi], D fixed 2
    # X* = [4.701055751981055, 3.152946019601391] or [-1.582142172055011,-3.130246799635430]
    # F* = -106.764
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1-X2)**2 + np.sin(X1)*np.e**((1-np.cos(X2))**2) + np.cos(X2)*np.e**((1-np.sin(X1))**2)
    
    return F

def Bohachevsky1(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1) - 0.4*np.cos(4*np.pi*X2) + 0.7
    
    return F

def Bohachevsky2(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1)*np.cos(4*np.pi*X2) + 0.3
    
    return F

def Bohachevsky3(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1+4*np.pi*X2)*np.cos(4*np.pi*X2) + 0.3
    
    return F

def Booth(X):
    # X in [-10, 10], D fixed 2
    # X* = [1, 3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+2*X2-7)**2 + (2*X1+X2-5)**2
    
    return F

def Branin(X):
    # X in [(-5, 10), (0, 15)], D fixed 2
    # X* = [-pi, 12.275] or [pi, 2.275] or [9.42478, 2.475]
    # F* = 0.39788735772973816
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    first = ( X2 - 5.1*X1**2/(4*np.pi**2) + 5*X1/np.pi - 6 )**2
    second = 10 * (1 - 1/(8*np.pi)) * np.cos(X1)
    third  = 10
    F = first + second + third
    
    return F

def Brent(X):
    # X in [-10, 10], D fixed 2
    # X* = [-10, -10]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+10)**2 + (X2+10)**2 + np.exp(-X1**2-X2**2)
    
    return F

def Brown(X):
    # X in [-1, 4]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( X[:, :-1]**(2*(X[:, 1:]+1)) + X[:, 1:]**(2*(X[:, :-1]+1)), axis=1 )
    
    return F

def Bukin4(X):
    # X in [(-15,-5), (-3, 3)], D fixed 2
    # X* = [-10, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*X2**2 + 0.01*np.abs(X1+10)
    
    return F

def Bukin6(X):
    # X in [(-15,-5), (-3, 3)], D fixed 2
    # X* = [-10, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*np.sqrt(np.abs(X2-0.01*X1**2)) + 0.01*np.abs(X1+10)
    
    return F

def Chung_Reynolds(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( X**2, axis=1 )**2
    
    return F

def Cigar(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = X[:, 0]**2 + 1E6*np.sum(X[:, 1:]**2, axis=1)
    
    return F

def Colville(X):
    # Wood
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

def Cosine_Mixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = -0.1*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1) - 0.1*np.sum(np.cos(5*np.pi*X), axis=1)
    
    return F

def Cross_in_Tray(X):
    # X in [-10, 10], D fixed 2
    # X* = [±1.349406685353340, ±1.349406608602084]
    # F* = -2.06261218
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1E-4*( np.abs( np.sin(X1)*np.sin(X2)*np.exp( np.abs(100-(X1**2+X2**2)**0.5/np.pi) ) ) + 1 )**0.1
    
    return F

def Csendes(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    
    check = np.prod(X, axis=1)
    mask = check!=0
    F[mask] = np.sum( X[mask]**6 * (2+np.sin(1/X[mask])) ,axis=1 )
    
    return F

def De_Jong3(X):
    # X in [-5.12, 5.12]
    # X* = [(-5.12, -5), (-5.12, -5), ..., (-5.12, -5)]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum( np.floor(X) , axis=1 ) + 6*D
    
    return F

def De_Jong5(X):
    # Shekel's Foxholes
    # X in [-65.536, 65.536], D fixed 2
    # X* = [-31.97833, -31.97833]
    # F* = 0.998003837794449325873406851315
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(25) + 1
    a1 = np.tile(np.array([-32, -16, 0, 16, 32]), 5)
    a2 = np.repeat(np.array([-32, -16, 0, 16, 32]), 5)
    
    for i in range(P):
        F[i] = 1/500 + np.sum( 1 / ( L + (X1[i]-a1)**6 + (X2[i]-a2)**6 ), axis=0 )
        
    F = 1 / F
    
    return F

def Deckkers_Aarts(X):
    # X in [-20, 20], D fixed 2
    # X* = [0, ±15]
    # F* = -24771.09375
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 1E5*X1**2 + X2**2 - (X1**2+X2**2)**2 + 1E-5*(X1**2+X2**2)**4
    
    return F

def Dixon_Price(X):
    # X in [-10, 10]
    # X* = [2**((2-2*1)/(2*1)), 2**((2-2*2)/(2*2)), ..., 2**((2-2*D)/(2*D))]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = (X[:, 0]-1)**2 + np.sum( L[1:]*(2*X[:, 1:]**2 - X[:, :-1])**2, axis=1 )
    
    return F

def Drop_wave(X):
    # X in [-5.12, 5.12], D fixed 2
    # X* = [0, 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1 * (1+np.cos(12*(X1**2+X2**2)**0.5)) / (0.5*(X1**2+X2**2)+2)
    
    return F

def Easom(X):
    # X in [-100, 100], D fixed 2
    # X* = [pi, pi]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1*np.cos(X1)*np.cos(X2)*np.exp(-1*(X1-np.pi)**2-(X2-np.pi)**2)
    
    return F

def Egg_Crate(X):
    # X in [-5, 5], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + X2**2 + 25*(np.sin(X1)**2+np.sin(X2)**2)
    
    return F

def Eggholder(X):
    # X in [-512, 512], D fixed 2
    # X* = [512, 404.2319]
    # F* = -959.6407
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -(X2+47) * np.sin(np.abs(X2 +X1/2+47)**0.5) - X1*np.sin(np.abs(X1-(X2+47))**0.5)
    
    return F

def Ellipsoid(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum(( X*1000**(L/(D-1)) )**2, axis=1)
    
    return F

def Ellipsoidal(X):
    # X in [-D, D]
    # X* = [0, 1, ..., D-1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum( (X-L)**2, axis=1 )
    
    return F

def Elliptic(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum( X**2 * 1E6**(L/(D-1)), axis=-1 )
    
    return F

def Exponential(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -1 * np.exp( -0.5 * np.sum(X**2, axis=1) )
    
    return F

def Five_well_potential(X):
    # X in [-20, 20], D fixed 2
    # X* = [4.92, -9.89]
    # F* = -1.4616
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    first = 1 + 0.05*(X1**2+(X2-10)**2)
    second = 1 + 0.05*((X1-10)**2+X2**2)
    third = 1 + 0.03*((X1+10)**2+X2**2)
    fourth = 1 + 0.05*((X1-5)**2+(X2+10)**2)
    fiveth = 1 + 0.1*((X1+5)**2+(X2+10)**2)
    
    F = (1 - 1/first - 1/second - 1.5/third - 2/fourth - 1/fiveth) * (1+1E-4*(X1**2+X2**2)**1.2)
    
    return F

def Foresster(X):
    # X in [0, 1], D fixed 1
    # X* = [0.76]
    # F* = [-6.01667]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    
    F = (6*X1-2)**2 * np.sin(12*X1-4)
    
    return F

def Goldstein_Price(X):
    # X in [-2, 2], D fixed 2
    # X* = [0, -1]
    # F* = 3
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F1 = 1 + (X1+X2+1)**2 * (19-14*X1+3*X1**2-14*X2+6*X1*X2+3*X2**2)
    F2 = 30 + (2*X1-3*X2)**2 * (18-32*X1+12*X1**2+48*X2-36*X1*X2+27*X2**2)
    
    F = F1*F2
    
    return F

def Gramacy_Lee(X):
    # X in [-0.5, 2.5], D fixed 1
    # X* = [0.548563444114526]
    # F* = -0.869011134989500
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    
    F = np.sin(10*np.pi*X1)/(2*X1) + (X1-1)**4
    
    return F

def Griewank(X):
    # X in [-600, 600]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = 1 + 1/4000 * np.sum(X**2, axis=1) - np.prod( np.cos(X/L), axis=1 )
    
    return F

def Happy_Cat(X):
    # X in [-2, 2]
    # X* = [-1, -1, ..., -1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    a = 1/8
    L1 = np.linalg.norm(X, ord=1, axis=1)
    F = ( ( L1-D )**2 )**a + ( 0.5*L1 + np.sum(X, axis=1) )/D + 1/2
    
    return F

def Hartmann3(X):
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

def Hartmann4(X):
    # X in [0, 1], D fixed 4
    # X* = [0.1873, 0.1906, 0.5566, 0.2647]
    # F* = -3.135474
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    a = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5],
                  [0.05, 10, 17, 0.1],
                  [3, 3.5, 1.7, 10],
                  [17, 8, 0.05, 10]])
    p = 1e-4 * np.array([[1312, 1696, 5569, 124],
                         [2329, 4135, 8307, 3736],
                         [2348, 1451, 3522, 2883],
                         [4047, 8828, 8732, 5743]])
    
    for k in range(P):
        first = a[0] * np.exp(-1*np.sum(A[0]*(X[k]-p[0])**2))
        second = a[1] * np.exp(-1*np.sum(A[1]*(X[k]-p[1])**2))
        third = a[2] * np.exp(-1*np.sum(A[2]*(X[k]-p[2])**2))
        fourth = a[3] * np.exp(-1*np.sum(A[3]*(X[k]-p[3])**2))
        
        F[k] = 1/0.839 * ( 1.1 - 1*(first + second + third + fourth) )
            
    
    return F

def Hartmann6(X):
    # X in [0, 1], D fixed 6
    # X* = [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]
    # F* = -3.32236801141551
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    a = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    p = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    
    for k in range(P):
        first = a[0] * np.exp(-1*np.sum(A[0]*(X[k]-p[0])**2))
        second = a[1] * np.exp(-1*np.sum(A[1]*(X[k]-p[1])**2))
        third = a[2] * np.exp(-1*np.sum(A[2]*(X[k]-p[2])**2))
        fourth = a[3] * np.exp(-1*np.sum(A[3]*(X[k]-p[3])**2))
        
        F[k] = -1*(first + second + third + fourth)
            
    
    return F

def Himmelblau(X):
    # X in [-6, 6], D fixed 2
    # X* = [3, 2] or [3.584428340330,-1.848126526964] or [-2.805118086953,3.131312518250] or [-3.779310253378, -3.283185991286]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+X2**2-7)**2 + (X1**2+X2-11)**2
    
    return F

def Holder_Table(X):
    # X in [-10, 10], D fixed 2
    # X* = [±8.05502, ±9.66459]
    # F* = -19.2085
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1 * np.abs( np.sin(X1)*np.cos(X2) * np.exp( np.abs( 1 - (X1**2+X2**2)**0.5/np.pi ) ) )
    
    return F

def Inverted_Cosine_Mixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 0.1*D - ( 0.1*np.sum(np.cos(5*np.pi*X), axis=1) - np.sum(X**2, axis=1) )
    
    return F

def k_tablet(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1) + np.sum( (100*X)**2, axis=1)
    
    return F

def Keane(X):
    # X in [0, 10], D fixed 2
    # X* = [1.393249070031784, 0] or [0, 1.393249070031784]
    # F* = 0.673667521146855
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1*np.sin(X1-X2)**2*np.sin(X1+X2)**2/(X1**2+X2**2)**0.5
    
    return F

def Kowalik(X):
    # X in [-5, 5], D fixed 4
    # X* = [0.192833, 0.190836, 0.123117, 0.135766]
    # F* = 0.0003
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = np.array([(4, 2, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16)])
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    for i in range(P):
        F[i] = np.sum( ( a - X1[i]*(b**2+b*X2[i])/(b**2+b*X3[i]+X4[i]) )**2, axis=1 )
    
    return F

def Langermann(X):
    # X in [0, 10], D fixed 2
    # X* = [?, ?]
    # F* = -4.15581
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    c = np.array([1, 2, 5, 2, 3])
    A = np.array([[3, 5],
                  [5, 2],
                  [2, 1],
                  [1, 4],
                  [7, 9]])
    F = np.zeros(P)
    
    for i in range(5):
        F = F + c[i] * np.exp(-1/np.pi*np.sum((X-A[i])**2, axis=1)*np.cos(np.pi*np.sum((X-A[i])**2, axis=1)))
    
    return F

def Leon(X):
    # X in [-1.2, 1.2], D fixed 2
    # X* = [1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*(X2-X1**3)**2 + (1-X1)**2
    
    return F

def Levi13(X):
    # X in [-10, 10], D fixed 2
    # X* = [1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.sin(3*np.pi*X1)**2 + (X1-1)**2*(1+np.sin(3*np.pi*X2)**2) + (X2-1)**2*(1+np.sin(2*np.pi*X2)**2)
    
    return F

def Levy_and_Montalvo_1(X):
    # LEVY
    # X in [-10, 10]
    # X* = [-1, -1, ..., -1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    y = 1 + 1/4 * (X+1)
    first = 10 * np.sin( np.pi*y[:, 0] )**2
    second = np.sum( (y[:, :-1]-1)**2 * ( 1 + 10*np.sin(np.pi*y[:, 1:])**2 ), axis=1 )
    third = ( y[:, -1] - 1 )**2
    
    F = np.pi/D * (first+second+third)
    
    return F

def Levy_and_Montalvo_2(X):
    # LEVY FUNCTION N. 13
    # X in [-5, 5]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    first = np.sin(3*np.pi*X[:, 0])**2
    second = np.sum( (X[:, :-1]-1)**2 * ( 1+np.sin(3*np.pi*X[:, 1:])**2 ), axis=1 )
    third = ( X[:, -1] - 1 )**2 * (1 + np.sin(2*np.pi*X[:, -1])**2)
    
    F = 0.1 * (first+second+third)
    
    return F

def Matyas(X):
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = 0.26*(X1**2+X2**2)-0.48*X1*X1
    
    return F

def McCormick(X):
    # X in [(-1.5, 4), (-3, 3)], D fixed 2
    # X* = [-0.547, -1.547]
    # F* = -1.9133
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = np.sin(X1+X2) + (X1-X2)**2 - 1.5*X1 + 2.5*X2 + 1
    
    return F

def Michalewicz(X, m=10):
    # X in [0, pi]
    # X*(D=2) = [2.20, 1.57], X*(D=5) = ?, X*(D=10) = ?
    # F*(D=2)=-1.8013 , F*(D=5)=-4.687658, F*(D=10)=-9.66015
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    L = np.arange(D) + 1
    F = -1*np.sum( np.sin(X)*np.sin(L*X**2/np.pi)**(2*m) , axis=1 )
    
    return F

def Modified_Double_Sum(X):
    # X in [-10.24, 10.24]
    # X* = [1, 2, ..., D]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( np.cumsum((X-L)**2, axis=1), axis=1)
    
    return F

def Modified_Foresster(X):
    # X in [0, 1], D fixed 1
    # X* = [0.092]
    # F* = [0.665113]
    if X.ndim==1:
        X = X.reshape(1, -1)
    A = 0.5
    B = 10
    C = -5
    
    X1 = X[:, 0]
    
    F = (6*X1-2)**2 * np.sin(12*X1-4)
    F = A*F + B*(X1-0.5) - C
    
    return F

def Moved_Axis_Parallel_Hyper_Ellipsoid(X):
    # X in [-500, 500]
    # X* = [5, 10, ..., 5D]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum(L*(X-5*L)**2, axis=1)
    
    return F

def Noncontinuous_Rastrigin(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    mask = np.abs(X)>=0.5
    X[mask] = np.round(2*X[mask])/2
    
    F = 10*D + np.sum( X**2 - 10*np.cos(2*np.pi*X), axis=1 )
    
    return F

def Paviani(X):
    # X in [2, 10], D fixed 10
    # X* = [9.351, 9.351, ..., 9.351]
    # F* = -45.77848
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.log( X-2 )**2 + np.log( 10-X )**2, axis=1 ) - np.prod(X, axis=1)**0.2
    
    return F

def Penalized1(X):
    # X in [-50, 50]
    # X* = [-1, -1, ..., -1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    y = 1 + 1/4 * (X+1)
    first = 10 * np.sin( np.pi*y[:, 0] )**2
    second = np.sum( (y[:, :-1]-1)**2 * ( 1 + 10*np.sin(np.pi*y[:, 1:])**2 ), axis=1 )
    third = ( y[:, -1] - 1 )**2
    
    F = np.pi/D * (first+second+third) + u(X, 10, 100, 4)
    
    return F

def Penalized2(X):
    # X in [-50, 50]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    first = np.sin(3*np.pi*X[:, 0])**2
    second = np.sum( (X[:, :-1]-1)**2 * ( 1+np.sin(3*np.pi*X[:, 1:])**2 ), axis=1 )
    third = ( X[:, -1] - 1 )**2 * (1 + np.sin(2*np.pi*X[:, -1])**2)
    
    F = 0.1 * (first+second+third) + u(X, 5, 100, 4)
    
    return F

def Periodic(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0.9
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 1 + np.sum(np.sin(X)**2, axis=1) -0.1*np.exp(np.sum(X**2, axis=1))
    
    return F

def Perm1(X):
    # X in [-D, D]
    # X* = [1, 2, ..., D]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    L = np.arange(D) + 1
    b = 0.5
    F = np.zeros(P)
    
    for i in range(D):
        F = F + np.sum( (L**(i+1)+b) * ((X/L)**(i+1)-1), axis=1 )**2
    
    return F

def Perm2(X):
    # X in [-D, D]
    # X* = [1/1, 1/2, ..., 1/D]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    L = np.arange(D) + 1
    b = 0.5
    F = np.zeros(P)
    
    for i in range(D):
        F = F + np.sum( (L+b) * (X-1/L), axis=1 )**2
    
    return F

def Powell(X):
    # X in [-4, 5]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    X1 = X[:, 0:int(D)//4*4:4]
    X2 = X[:, 1:int(D)//4*4:4]
    X3 = X[:, 2:int(D)//4*4:4]
    X4 = X[:, 3:int(D)//4*4:4]
    
    F = np.sum( (X1+10*X2)**2 + 5*(X3-X4)**2 + (X2-2*X3)**4 + 10*(X1-X4)**4, axis=1 )
    
    return F

def Powell_sum(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( np.abs(X)**L, axis=1 )
    
    return F

def Power_Sum(X):
    # X in [0, D], D fixed 4
    # X* = [1, 2, 2, 3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    b = np.array([8, 18, 44, 114])
    F = np.zeros(P)
    L = np.arange(D) + 1
    
    for i in range(D):
        F = F + ( b[i] - np.sum( X**L[i] , axis=1) )**2
    
    return F

def Qing(X):
    # X in [-500, 500]
    # X* = [1, 2**0.5, 3**0.5 ..., D**0.5]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( X**2-L , axis=1 )
    
    return F

def Quartic(X, with_noise=False):
    # modified De_Jong4
    # X in [-1.28, 1.28]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    
    if with_noise==True:
        r = np.random.uniform(size=[P])
    else:
        r = 0
    L = np.arange(D)
    
    F = np.sum( L*X**4, axis=1 ) + r
    
    return F

def Rana(X):
    # X in [-512, 512], D fixed 2
    # X* = [-488.6326, 512]
    # F* = -511.73
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1 * np.sin((np.abs(X2+1-X1))**0.5) * np.cos((np.abs(X1+X2+1))**0.5) + (X2+1)*np.cos((X2+1-X1)**0.5) * np.sin((np.abs(X1+X2+1))**0.5)
    
    return F

def Rastrigin(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 10*D + np.sum( X**2 - 10*np.cos(2*np.pi*X), axis=1 )
    
    return F

def Ridge(X):
    # not sure
    # X in [-5, 5]
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    d = 1
    a = 0.5
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1 + d*( np.sum( X2**2 ) )**a
    
    return F

def Rosenbrock(X):
    # De_Jong2, Banana
    # X in [-30, 30]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( 100*(X[:, 1:]-X[:, :-1]**2)**2 + (X[:, :-1]-1)**2, axis=1 )
    
    return F

def Salomon(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 1 - np.cos(2*np.pi*np.sum(X**2, axis=1)**0.5 + 0.1*np.sum(X**2, axis=1)**0.5)
    
    return F

def Schaffer1(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + (np.sin((X1**2+X2**2)**2)**2-0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F

def Schaffer2(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + ((np.sin(X1**2-X2**2))**2 - 0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F

def Schaffer3(X):
    # X in [-100, 100], D fixed 2
    # X* = [±1.253114962205510, 0] or [0, ±1.253114962205510]
    # F* = 0.001566854526004
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + ((np.sin(np.cos(np.abs(X1**2-X2**2))))**2 - 0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F

def Schaffer4(X):
    # X in [-100, 100], D fixed 2
    # X* = [±1.253114962205510, 0] or [0, ±1.253114962205510]
    # F* = 0.292578632035980
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + ((np.cos(np.sin(np.abs(X1**2-X2**2))))**2 - 0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F
	
def Schaffer6(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + (np.sin((X1**2+X2**2)**0.5)**2-0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F

def Schaffer7(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    si = ( (X[:, :-1]**2 + X[:, 1:]**2)**0.5 ).flatten()
    F = 1/(D-1) * si**0.5 * (np.sin(50*si**0.2))**2
    
    return F

def Schwefel221(X):
    # Schwefel 2.21, MaxMod
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.max(np.abs(X), axis=1)
    
    return F

def Schwefel222(X):
    # Schwefel 2.22
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)
    
    return F

def Schwefel12(X):
    # Schwefel 1.2, Rotated Hyper-Ellipsoid, Double-Sum
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.cumsum(X, axis=1), axis=1)
    
    return F

def Schwefel223(X):
    # Schwefel 2.23
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**10, axis=1 )
    
    return F

def Schwefel226(X):
    # Schwefel 2.26
    # X in [-500, 500]
    # X* = [420.96874636, 420.96874636, ..., 420.96874636]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 418.9828872724339*D - np.sum(X * np.sin( np.abs(X)**0.5 ), axis=1 )
    
    return F

def Schwefel6(X):
    # Schwefel 2.20
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.abs(X), axis=1 )
    
    return F

def Shekel(X, m=5):
    # X in [0, 10], D fixed 4
    # X* = [4, 4, 4, 4]
    # F*(m=5)=-10.1532 , F*(m=7)=-10.4029, F*(m=10)=-10.5364
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                  [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
    
    F = np.zeros([P])
    for i in range(m):
        F = F + 1/(np.sum( (X-C[:, i])**2, axis=1 )  + b[i])
    
    F = -1*F
    
    return F

def Shubert(X):
    # X in [-10, 10], D fixed 2
    # X* = [-7.0835, 4.8580] or [-7.0835,-7.7083] or
    #      [-1.4251,-7.0835] or [ 5.4828, 4.8580] or
    #      [-1.4251,-0.8003] or [ 4.8580, 5.4828] or
    #      [-7.7083,-7.0835] or [-7.0835,-1.4251] or
    #      [-7.7083,-0.8003] or [-7.7083, 5.4828] or
    #      [-0.8003,-7.7083] or [-0.8003,-1.4251] or
    #      [-0.8003, 4.8580] or [-1.4251, 5.4828] or
    #      [ 5.4828,-7.7083] or [ 4.8580,-7.0835] or
    #      [ 5.4828,-1.4251] or [ 4.8580,-0.8003]
    # F* = -186.7309
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(5) + 1
    F1 = np.cos( (L+1)*X1 + L )
    F2 = np.cos( (L+1)*X2 + L )
    
    F = F1*F2
    
    return F

def Shubert3(X):
    # X in [-10, 10], D fixed 2
    # X* = ?
    # F* = -29.6733337
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(5) + 1
    F1 = (L+1)*np.sin( (L+1)*X1 + L )
    F2 = (L+1)*np.sin( (L+1)*X2 + L )
    
    F = F1*F2
    
    return F

def Shubert4(X):
    # X in [-10, 10], D fixed 2
    # X* = ?
    # F* = -25.740858
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(5) + 1
    F1 = (L+1)*np.cos( (L+1)*X1 + L )
    F2 = (L+1)*np.cos( (L+1)*X2 + L )
    
    F = F1*F2
    
    return F

def Sinusoidal(X):
    # X in [0, pi]
    # X* = (2/3) * [pi, pi, ..., pi]
    # F* = -3.5
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -1 * ( 2.5*np.prod(np.sin(X-np.pi/6), axis=1) + np.prod(np.sin(5*(X-np.pi/6)), axis=1) )
    
    return F

def Six_Hump_Camel_Back(X):
    # X in [-5, 5], D fixed 2
    # X* = (±0.08984201368301331,±0.7126564032704135)
    # F* = -1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + 1/3*X1**6 + X1*X2 - 4*X2**2 + 4*X2**4
    
    return F

def Sphere(X):
    # Spherical Contours, Square Sum, Harmonic, De_Jong1 or Schumer-Steiglitz1
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1)
    
    return F

def Step1(X):
    # X in [-100, 100]
    # X* = -1<X<1
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.floor(np.abs(X)), axis=1)
    
    return F

def Step2(X):
    # X in [-100, 100]
    # X* = -0.5<=X<0.5
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.floor(np.abs(X+0.5))**2, axis=1)
    
    return F

def Step3(X):
    # X in [-100, 100]
    # X* = -1<X<1
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.floor(X**2), axis=1)
    
    return F

def Stepint(X):
    # X in [-5.12, 5.12]
    # X* = -5.12<X<-5
    # F* = 25-6*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 25 + np.sum(np.floor(X), axis=1)
    
    return F

def Styblinski_Tang(X):
    # X in [-5, 5]
    # X* = [-2.903534, -2.903534, ..., -2.903534]
    # F* = -39.16599*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 0.5 * np.sum( X**4-16*X**2+5*X , axis=1 )
    
    return F

def Sum_of_different_power(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( np.abs(X)**(L+1) , axis=1)
    
    return F

def Sum_Squares(X):
    # Axis parallel hyper-ellipsoid, Weighted Sphere, hyper ellipsodic
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum(L*X**2, axis=1)
    
    return F

def Tablet(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 1e6*X[:, 0] + np.sum( X[:, 1:]**6, axis=1 )
    
    return F

def Three_Hump_Camel(X):
    # X in [-5, 5], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 2*X1 - 1.05*X2**4 + X1**6/6 + X1*X2 + X2**2
    
    return F

def Trid(X):
    # X in [-D**2, D**2]
    # X* = [1*(D+1-1), 2*(D+1-2), ..., D*(D+1-D)]
    # F* = -D*(D+4)*(D-1)/6
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum((X-1)**2, axis=1) - np.sum(X[:, 1:]*X[:, :-1], axis=1)
    
    return F

def Whitley(X):
    # X in [-10.24, 10.24]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    F = np.zeros(P)
    
    for i in range(D):
        Xi = X[:, i].reshape(-1, 1)
        F = F + np.sum( (100*(Xi**2-X)**2 + (1-X)**2)**2/4000 - np.cos(100*(Xi**2-X)**2+(1-X)**2) + 1, axis=1 )
    
    return F

def Wolfe(X):
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

def Xin_She_Yang1(X):
    # X in [-5, 5]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    L = np.arange(D) + 1
    r = np.random.uniform(size=[P, D])
    
    F = np.sum( r*np.abs(X)**L , axis=1 )
    
    return F

def Xin_She_Yang2(X):
    # X in [-2pi, 2pi]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.abs(X) , axis=1) * np.exp( -1*np.sum( np.sin(X**2) , axis=1 ) )
    
    return F

def Xin_She_Yang3(X):
    # X in [-2pi, 2pi]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    m = 5
    b = 15
    
    F = np.exp( -1*np.sum( (X/b)**(2*m), axis=1 ) ) - 2*np.exp( -1*np.sum( X**2, axis=1 ) ) * np.prod(np.cos(X)**2, axis=1)
    
    return F

def Xin_She_Yang4(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = ( np.sum( np.sin(X)**2 , axis=1 ) - np.exp(-1*np.sum(X**2, axis=1)) ) * np.exp(-1*np.sum( np.sin(np.abs(X)**0.5)**2 , axis=1 ))
    
    return F

def Zakharov(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum(X**2, axis=1) + np.sum(L/2*X, axis=1)**2 + np.sum(L/2*X, axis=1)**4
    
    return F

def Zettl(X):
    # X in [-1, 5], D fixed 2
    # X* = [-0.02989597760285287, 0]
    # F* = -0.003791237220468656
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.25*X1 + ( X1**2 - 2*X1 + X2**2 )**2
    
    return F

def u(X, a, k, m):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    
    F = np.zeros([P, D])
    mask1 = X>a
    mask3 = X<-a
    mask2 = ~(mask1+mask3)
    
    F[mask1] = k*(X[mask1]-a)**m
    F[mask3] = k*(-X[mask3]-a)**m
    F[mask2] = 0
    
    return F.sum(axis=1)