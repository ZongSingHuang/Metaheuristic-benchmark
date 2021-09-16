# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

import numpy as np

def Sphere(X):
    # X in [-100, 100]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1)
    
    return F

def Rastrigin(X):
    # X in [-5.12, 5.12]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2 - 10*np.cos(2*np.pi) + 10, axis=1)
    
    return F

def Ackley(X):
    # X in [-32, 32]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    f1 = -0.2*np.sqrt( np.sum(X**2, axis=1)/D )
    f2 = np.sum( np.cos(2*np.pi*X), axis=1 )/D
    
    F = -20*np.exp(f1) - np.exp(f2) + 20 +np.e
    
    return F

def Griewank(X):
    # X in [-600, 600]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    
    f1 = np.sum(X**2, axis=1)
    f2 = np.prod( np.cos(X/np.sqrt(i)), axis=1 )
    
    F = 1/4000 * f1 - f2 + 1
    
    return F

def Schwefel_P222(X):
    # X in [-10, 10]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.abs(X), axis=1 ) + np.prod( np.abs(X), axis=1 )
    
    return F

def Rosenbrock(X):
    # X in [-30, 30]
    # F* = 0
    # X* = [1, 1, ..., 1]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    f1 = 100 * (X[:, 1:]-X[:, :-1]**2)**2
    f2 = (X[:, :-1]-1)**2
    
    F = np.sum(f1+f2, axis=1)
    
    return F

def Sehwwefel_P221(X):
    # X in [-100, 100]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.max( np.abs(X) , axis=1 )
    
    return F

def Quartic(X):
    # X in [-1.28, 1.28]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    i = np.arange(D) + 1
    
    F = np.sum( i*X**4 , axis=1 ) + np.random.uniform(size=[P])
    
    return F

def Schwefel_P12(X):
    # X in [-100, 100]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)

    F = np.sum( np.cumsum(X, axis=1) , axis=1 )
    
    return F

def Penalized1(X):
    # X in [-50, 50]
    # F* = 0
    # X* = [-1, -1, ..., -1]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    y1 = y(X[:, 0])
    yD = y(X[:, -1])
    yi = y(X[:, :-1])
    yi_1 = y(X[:, 1:])
    
    f1 = 10 * np.sin(np.pi*y1)**2
    f2 = np.sum( (yi-1)**2 * (1+10*np.sin(np.pi*yi_1)**2), axis=1 )
    f3 = (yD - 1)**2
    
    F = np.pi/D * (f1+f2+f3) + u(X, 10, 100, 4)
    
    return F

def Penalized2(X):
    # X in [-50, 50]
    # F* = 0
    # X* = [1, 1, ..., 1]
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    XD = X[:, -1]
    Xi = X[:, :-1]
    Xi_1 = X[:, 1:]
    
    f1 = np.sin(3*np.pi*X1)**2
    f2 = np.sum( (Xi-1)**2 * (1+np.sin(3*np.pi*Xi_1)**2), axis=1 )
    f3 = (XD-1)**2 * (1+np.sin(2*np.pi*XD)**2)
    
    F = 0.1 * (f1+f2+f3) + u(X, 5, 100, 4)
    
    return F

def Schwefel_226(X):
    # X in [-500, 500]
    # F* = -418.982887272433799807913601398D
    # X* = [420.968746, 420.968746, ..., 420.968746]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -np.sum( X * np.sin( np.sqrt( np.abs(X) ) ), axis=1 )
    
    return F

def Step(X):
    # X in [-100, 100]
    # F* = 0
    # X* in [-0.5, 0.5)
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.floor( np.abs(X+0.5) )**2 , axis=1 )
    
    return F

def Kowalik(X):
    # X in [-5, 5], D fixed 4
    # F* = 0.00030748610
    # X* = [0.192833, 0.190836, 0.123117, 0.135766]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = np.array([4, 2, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16])
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    
    for i in range(P):
        f1 = X1[i]*(b**2+b*X2[i])
        f2 = b**2+b*X3[i]+X4[i]
        F[i] = np.sum( (a - f1/f2)**2 )
        
    return F

def ShekelFoxholes(X):
    # X in [-65.536, 65.536], D fixed 2
    # F* = 0.998003837794449325873406851315
    # X* = [-31.97833, -31.97833]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    j = np.arange(25) + 1
    a1 = np.tile(np.array([-32, -16, 0, 16, 32]), 5)
    a2 = np.repeat(np.array([-32, -16, 0, 16, 32]), 5)
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    for i in range(P):
        f1 = j + (X1[i]-a1)**6 + (X2[i]-a2)**6
        F[i] = ( 1/500 + np.sum( 1/f1 ) )**-1
        
    return F

def GoldsteinPrice(X):
    # X in [-2, 2], D fixed 2
    # F* = 3
    # X* = [0, -1]
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = 1 + (X1+X2+1)**2 * (19-14*X1+3*X1**2-14*X2+6*X1*X2+3*X2**2)
    f2 = 30 + (2*X1-3*X2)**2 * (18-32*X1+12*X1**2+48*X2-36*X1*X2+27*X2**2)
    
    F = f1 * f2

    return F

def Shekel(X, m=5):
    # X in [0, 10], D fixed 4
    # F*(m=5) = -10.1532 , F*(m=7) = -10.4029, F*(m=10) = -10.5364
    # X* = [4, 4, 4, 4]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    a = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                  [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
    c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    
    for i in range(m):
        f1 = np.sum( (X-a[:, i])**2, axis=1 )  + c[i]
        F = F + 1/f1

    return -F

def Branin(X):
    # X1 in [-5, 10], X2 in [0, 15], D fixed 2
    # F* = 0.39788735772973816
    # X* = [-PI, 12.275], [PI, 2.275], [9.42478, 2.475]
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = ( X2 - 5.1*X1**2/(4*np.pi**2) + 5*X1/np.pi - 6 )**2
    f2 = 10 * ( 1 - 1/(8*np.pi) ) * np.cos(X1)
    F = f1 + f2 + 10
    
    return F

def Hartmann3(X):
    # X in [0, 1], D fixed 3
    # F* = -3.86278214782076
    # X* = [0.1, 0.55592003, 0.85218259]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    m = 4
    a = np.array([[3, 10, 30],
                  [0.1, 10, 35],
                  [3, 10, 30],
                  [0.1, 10, 35]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.36890, 0.1170, 0.2673],
                  [0.46990, 0.4387, 0.7470],
                  [0.10910, 0.8732, 0.5547],
                  [0.03815, 0.5743, 0.8828]])
    
    for i in range(m):
        f1 = -np.sum( a[i]*(X-p[i])**2, axis=1 )
        F = F + c[i] * np.exp(f1)
    
    return -F

def SixHumpCamelBack(X):
    # X in [-5, 5], D fixed 2
    # F* = -1.031628453489877
    # X* = [-0.08984201368301331, 0.7126564032704135], [0.08984201368301331, -0.7126564032704135]
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 -2.1*X1**4 + X1**6/3 + X1*X2 -4*X2**2 + 4*X2**4
    
    return F

def Zakharov(X):
    # X in [-5, 10]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    F = np.sum(X**2, axis=1) + np.sum(0.5*i*X, axis=1)**2 + np.sum(0.5*i*X, axis=1)**4
    
    return F

def SumSquares(X):
    # X in [-10, 10]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    
    F = np.sum(i*X**2, axis=1)
    
    return F

def Alpine(X):
    # X in [-10, 10]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)

    F = np.sum( np.abs( X*np.sin(X) + 0.1*X ) , axis=1)
    
    return F

def Michalewicz(X, m=10):
    # X in [0, PI]
    # F*(D=1) = -0.801303410098552549 , F*(D=2) = -1.80130341009855321, F*(D=5) = -4.687658, F*(D=10) = -9.66015
    # X*(D=1) = 2.20290552017261 , X*(D=2) = [2.20290552014618, 1.57079632677565]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    f1 = np.sin(X)
    f2 = np.sin( i*X**2/np.pi )**(2*m)
    
    F = -np.sum( f1*f2 , axis=1)
    
    return F

def Exponential(X):
    # X in [-1, 1]
    # F* = 1
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.exp( -0.5 * np.sum( X**2, axis=1 ) )
    
    return F

def Schaffer(X):
    # X in [-100, 100], D fixed 2
    # F* = 0
    # X* = [0, 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( np.sqrt( X1**2+X2**2 ) )**2 - 0.5
    f2 = ( 1 + 0.001*(X1**2+X2**2) )**2
    
    F = 0.5 - f1/f2
    
    return F

def BentCigar(X):
    # X in [-100, 100]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = X[:, 0]**2 + 1E6*np.sum(X[:, 1:]**2, axis=1)
    
    return F

def Bohachevsky1(X):
    # X in [-50, 50], D fixed 2
    # F* = 0
    # X* = [0, 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1) - np.cos(4*np.pi*X2) + 0.7
    
    return F

def Ellipsoid(X):
    # X in [-100, 100]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    f1 = (i-1)/D-1
    
    F = np.sum( 1E6**f1 * X**2, axis=1 )
    
    return F

def DropWave(X):
    # X in [-5.12, 5.12], D fixed 2
    # F* = -1
    # X* = [0, 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    X2 = X[:, 1]
    f1 = 1 + np.cos(12*np.sqrt(X1**2+X2**2))
    f2 = 0.5*(X1**2+X2**2) + 2
    
    F = f1/f2
    
    return -F

def CosineMixture(X):
    # X in [-1, 1]
    # F* = 0.1D
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    f1 = 0.1 * np.sum( np.cos(5*np.pi*X), axis=1 )
    f2 = np.sum(X, axis=1)
    
    F = f1 - f2
    
    return F

def Ellipsoidal(X):
    # X in [-D, D]
    # F* = 0
    # X* = [1, 2, ..., D]
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    D = X.shape[1]
    i = np.arange(D) + 1
    
    F = np.sum(X-i, axis=1)
    
    return F

def LevyandMontalvo1(X):
    # X in [-10, 10]
    # F* = 0
    # X* = [-1, -1, ..., -1]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    y1 = y(X[:, 0])
    yi = y(X[:, :-1])
    yi_1 = y(X[:, 1:])
    yD = y(X[:, -1])
    
    f1 = 10 * np.sin(np.pi*y1)**2
    f2 = np.sum( (yi-1)**2 * (1+10*np.sin(np.pi*yi_1)**2) , axis=1 )
    f3 = (yD - 1)**2
    
    F = np.pi/D * (f1 + f2 + f3)
    
    return F

def Easom(X):
    # X in [-10, 10], D fixed 2
    # F* = -1
    # X* = [PI, PI]
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    F = -np.cos(X1) * np.cos(X2) * np.exp(-(X1-np.pi)**2-(X2-np.pi)**2)
    
    return F

def PowellSum(X):
    # X in [-1, 1]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    i = np.arange(D) + 1
    
    F = np.sum( np.abs(X)**(i+1), axis=1 )
    
    return F

def LevyandMontalvo2(X):
    # X in [-5, 5]
    # F* = 0
    # X* = [1, 1, ..., 1]
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    XD = X[:, -1]
    
    f1 = np.sin(3*np.pi*X1)**2
    f2 = np.sum( (X-1)**2 * (1+np.sin(3*np.pi*X+1)) , axis=1 )
    f3 = (XD-1)**2 * (1+np.sin(2*np.pi*XD)**2)
    
    F = 0.1 * (f1 + f2 + f3)
    
    return F

def Holzman(X):
    # X in [-10, 10]
    # F* = 0
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)

    D = X.shape[1]
    i = np.arange(D) + 1
    
    F = np.sum( i*X**4, axis=1 )
    
    return F

def XinSheYang1(X):
    # X in [-20, 20]
    # F* = -1
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)

    m = 5
    beta = 15
    c = 0
    
    f1 = np.exp( -np.sum( (X/beta)**(2*m), axis=1 ) )
    f2 = 2 * np.exp( -np.sum( (X-c)**2 , axis=1 ) )
    f3 = np.prod( np.cos(X)**2, axis=1 )
    
    F = (f1 - f2) * f3
    
    return F

def XinSheYang6(X):
    # X in [-10, 10]
    # F* = -1
    # X* = [0, 0, ..., 0]
    if X.ndim==1:
        X = X.reshape(1, -1)

    f1 = np.sum( np.sin(X)**2, axis=1 )
    f2 = np.exp( -np.sum(X**2, axis=1) )
    f3 = np.exp( -np.sum( np.sin(np.sqrt(np.abs(X)))**2, axis=1 ) )
    
    F = (f1 - f2) * f3
    
    return F

def Beale(X):
    # X in [-4.5, 4.5], D fixed 2
    # F* = 0
    # X* = [3, 0.5]
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1.5-X1+X1*X2)**2 + (2.25-X1+X1*X2**2)**2 + (2.625-X1+X1*X2**3)**2
    
    return F

def Shubert(X):
    # X in [-10, 10], D fixed 2
    # F* = -186.7309
    # X* = [-7.0835, 4.8580],  [-7.0835, -7.7083], [-7.0835, -1.4251],
    #      [-7.7083, -7.0835], [-7.7083, 5.4828],  [-7.7083, -0.8003],
    #      [-1.4251, -7.0835], [-1.4251, -0.8003], [-1.4251, 5.4828],
    #      [4.8580, -7.0835],  [4.8580, 5.4828],   [4.8580, -0.8003],
    #      [5.4828, 4.8580],   [5.4828, -7.7083],  [5.4828, -1.4251],
    #      [-0.8003, -7.7083], [-0.8003, -1.4251], [-0.8003, 4.8580]
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    i = np.arange(5) + 1
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    for k in range(P):
        f1 = np.sum( i*np.cos( (i+1)*X1[k] + i ) )
        f2 = np.sum( i*np.cos( (i+1)*X2[k] + i ) )
    
        F[k] = f1 * f2
    
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