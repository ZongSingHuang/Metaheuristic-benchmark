# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

import numpy as np

def Sphere(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1)
    
    return F

def Rastrigin(X):
    # [1]
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( X**2 - 10*np.cos(2*np.pi*X) + 10, axis=1 )
    
    return F

def Ackley(X):
    # X in [-32, 32]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    f1 = -0.2 * ( np.sum(X**2, axis=1) / D )**0.5
    f2 = np.sum( np.cos(2*np.pi*X), axis=1 ) / D
    
    F = -20*np.exp(f1) - np.exp(f2) + 20 + np.exp(1)
    
    return F

def Griewank(X):
    # X in [-600, 600]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = (np.arange(D) + 1)**0.5
    
    F = 0.00025 * np.sum(X**2, axis=1) - np.prod( np.cos(X/L), axis=1 ) + 1
    
    return F

def Schwefel_P222(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)
    
    return F

def Rosenbrock(X):
    # X in [-30, 30]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( 100*(X[:, 1:]-X[:, :-1]**2)**2 + (X[:, :-1]-1)**2, axis=1 )
    
    return F

def Schwefel_P221(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.max(np.abs(X), axis=1)
    
    return F

def Quartic(X):
    # X in [-1.28, 1.28]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    
    r = np.random.uniform(size=[P])
    L = np.arange(D)
    
    F = np.sum( L*X**4, axis=1 ) + r
    
    return F

def Schwefel_P12(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.cumsum(X, axis=1), axis=1)
    
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

def Schwefel_P226(X):
    # X in [-500, 500]
    # X* = [420.96874636, 420.96874636, ..., 420.96874636]
    # F* = -418.982887272433799807913601398D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -np.sum( X * np.sin( np.abs(X)**0.5 ), axis=1 )
    
    return F

def Step(X):
    # X in [-100, 100]
    # X* = -0.5<=X<0.5
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.floor( np.abs(X+0.5) )**2, axis=1 )
    
    return F

def Kowalik(X):
    # X in [-5, 5], D fixed 4
    # X* = [0.192833, 0.190836, 0.123117, 0.135766]
    # F* = 0.00030748610
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

def ShekelFoxholes(X):
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

def GoldsteinPrice(X):
    # X in [-2, 2], D fixed 2
    # X* = [0, -1]
    # F* = 3
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = 1 + (X1+X2+1)**2 * (19-14*X1+3*X1**2-14*X2+6*X1*X2+3*X2**2)
    f2 = 30 + (2*X1-3*X2)**2 * (18-32*X1+12*X1**2+48*X2-36*X1*X2+27*X2**2)
    
    F = f1*f2
    
    return F

def Shekel(X, m=5):
    # X in [0, 10], D fixed 4
    # X* = [4, 4, 4, 4]
    # F*(m=5)=-10.1499 , F*(m=7)=-10.3999, F*(m=10)=-10.5319
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

def Branin(X):
    # X1 in [-5, 10], X2 in [0, 15], D fixed 2
    # X* = [-PI, 12.275], [PI, 2.275], [9.42478, 2.475]
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

def SixHumpCamelBack(X):
    # X in [-5, 5], D fixed 2
    # X* = [-0.08984201368301331, 0.7126564032704135], [0.08984201368301331, -0.7126564032704135]
    # F* = -1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + X1**6/3 + X1*X2 - 4*X2**2 + 4*X2**4
    
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

def Zakharov(X):
    # X in [-5, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum(X**2, axis=1) + np.sum(L/2*X, axis=1)**2 + np.sum(L/2*X, axis=1)**4
    
    return F

def SumSquares(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum(L*X**2, axis=1)
    
    return F

def Alpine(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.abs( X*np.sin(X) + 0.1*X ), axis=1 )
    
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

def Exponential(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.exp( -0.5 * np.sum(X**2, axis=1) )
    
    return F

def Schaffer(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( (X1**2+X2**2)**0.5 )**2 - 0.5
    f2 = ( 1 + 0.001*(X1**2+X2**2) )**2
    F = 0.5 + f1/f2
    
    return F

def BentCigar(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = X[:, 0]**2 + 1E6*np.sum(X[:, 1:]**2, axis=1)
    
    return F

def Bohachevsky1(X):
    # X in [-50, 50], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1) - 0.4*np.cos(4*np.pi*X2) + 0.7
    
    return F

def Ellipsoid(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( X**2 * 1E6**(L/(D-1)), axis=-1 )
    
    return F

def DropWave(X):
    # X in [-5.12, 5.12], D fixed 2
    # X* = [0, 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1 * (1+np.cos(12*(X1**2+X2**2)**0.5)) / (0.5*(X1**2+X2**2)+2)
    
    return F

def CosineMixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0.1*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 0.1*np.sum( np.cos(5*np.pi*X), axis=1 ) - np.sum(X**2, axis=1)
    
    return F

def Ellipsoidal(X):
    # X in [-D, D]
    # X* = [0, 1, ..., D-1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( (X-L)**2, axis=1 )
    
    return F

def LevyandMontalvo1(X):
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

def Easom(X):
    # X in [-10, 10], D fixed 2
    # X* = [pi, pi]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -np.cos(X1)*np.cos(X2)*np.exp(-(X1-np.pi)**2-(X2-np.pi)**2)
    
    return F

def PowellSum(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
    F = np.sum( np.abs(X)**(L+1) , axis=1)
    
    return F

def LevyandMontalvo2(X):
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

def Holzman(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    L = np.arange(D) + 1
    
    F = np.sum( L*X**4, axis=1 )
    
    return F

def XinSheYang1(X):
    # X in [-20, 20]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    m = 5
    b = 15
    
    F = np.exp( -1*np.sum( (X/b)**(2*m), axis=1 ) ) - 2*np.exp( -1*np.sum( X**2, axis=1 ) ) * np.prod(np.cos(X)**2, axis=1)
    
    return F

def XinSheYang6(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = ( np.sum( np.sin(X)**2 , axis=1 ) - np.exp(-1*np.sum(X**2, axis=1)) ) * np.exp(-1*np.sum( np.sin(np.abs(X)**0.5)**2 , axis=1 ))
    
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

def InvertedCosineMixture(X):
    # X in [-5, 5]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 0.1*D - ( 0.1*np.sum(np.cos(5*np.pi*X), axis=1) - np.sum(X**2, axis=1) )
    
    return F

def Salomon(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 1 - np.cos(2*np.pi*np.sum(X**2, axis=1)**0.5 + 0.1*np.sum(X**2, axis=1)**0.5)
    
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

def Paviani(X):
    # X in [2, 10], D fixed 10
    # X* = [9.35027, 9.35027, ..., 9.35027]
    # F* = -45.77848
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.log( X-2 )**2 + np.log( 10-X )**2, axis=1 ) - np.prod(X, axis=1)**0.2
    
    return F

def Sinusoidal(X):
    # X in [0, pi]
    # X* = (2/3) * [pi, pi, ..., pi]
    # F* = -3.5
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -1 * ( 2.5*np.prod(np.sin(X-np.pi/6), axis=1) + np.prod(np.sin(5*(X-np.pi/6)), axis=1) )
    
    return F

def ktablet(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1) + np.sum( (100*X)**2, axis=1)
    
    return F

def NoncontinuousRastrigin(X):
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

def Fletcher(X):
    return F

def Levy(X):
    # X in [-10, 10], D fixed 2
    # X* = [1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.sin(3*np.pi*X1)**2 + (X1-1)**2*(1+np.sin(3*np.pi*X2)**2) + (X2-1)**2*(1+np.sin(2*np.pi*X2)**2)
    
    return F

def Davis(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1**2+X2**2)**0.25 * ((np.sin(50*(3*X1**2+X2**2)**0.1))**2 + 1)
    
    return F

def Pathological(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( 0.5 + (np.sin((100*X[:, :-1]**2+X[:, 1:]**2)**0.5)**2 - 0.5)/(1+0.001*(X[:, :-1]**2-2*X[:, :-1]*X[:, 1:]+X[:, 1:]**2))**2, axis=1 )
    
    return F

def Schwefel220(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.abs(X), axis=1 )
    
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

def Tablet(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 1e6*X[:, 0] + np.sum( X[:, 1:]**6, axis=1 )
    
    return F

def Brown(X):
    # X in [-1, 4]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( (X[:, :-1]**2)**(X[:, 1:]**2+1) + (X[:, 1:]**2)**(X[:, :-1]**2+1), axis=1 )
    
    return F

def ChungReynolds(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( X**2, axis=1 )**2
    
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

def Bohachevsky2(X):
    # X in [-50, 50], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1)*np.cos(4*np.pi*X2) + 0.3
    
    return F

def Bohachevsky3(X):
    # X in [-50, 50], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1+4*np.pi*X2) + 0.3
    
    return F

def Colville(X):
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

def BartelsConn(X):
    # X in [-500, 500], D fixed 2
    # X* = [0, 0]
    # F* = 1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.abs(X1**2+X2**2+X1*X2) + np.abs(np.sin(X1)) + np.abs(np.cos(X2))
    
    return F

def Bird(X):
    # X in [-2PI, PI], D fixed 2
    # X* = [4.701055751981055, 3.152946019601391], [-1.582142172055011,-3.130246799635430]
    # F* = -106.7645367198034
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1-X2)**2 + np.sin(X1)*np.e**((1-np.cos(X2))**2) + np.cos(X2)*np.e**((1-np.sin(X1))**2)
    
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