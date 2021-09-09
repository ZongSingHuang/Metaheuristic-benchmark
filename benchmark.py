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
# n-D
# =============================================================================
def Cola(X):
    # [1]
    # X1 in [0, 4], other X in [-4, 4], D fixed 17
    # X* = [0.651906, 1.30194, 0.099242, -0.883791, -0.8796,
    #       0.204651, -3.28414, 0.851188, -3.46245, 2.53245, -0.895246,
    #       1.40992, -3.07367, 1.96257, -2.97872, -0.807849, -1.68978]
    # F* = 11.7464
    if X.ndim==1:
        X = X.reshape(1, -1)
            
    return F

















def Ackley1(X):
    # X in [-35, 35]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    left = -20 * np.exp( -0.02 * ( 1/D * np.sum(X**2, axis=1) )**0.5 )
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
    
    F = -200*np.exp(-0.02*(X1**2+X2**2)**0.5)
    
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



def Alpine1(X):
    # X in [-10, 10]
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















def Brown(X):
    # X in [-1, 4]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( (X[:, :-1]**2)**(X[:, 1:]**2+1) + (X[:, 1:]**2)**(X[:, :-1]**2+1), axis=1 )
    
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



def Cosine_Mixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = -0.1*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1) - 0.1*np.sum(np.cos(5*np.pi*X), axis=1)
    
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
    L = np.arange(D) + 1
    
    F = np.sum(( X*1000**(L/(D-1)) )**2, axis=1)
    
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

def Elliptic(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D) + 1
    
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
    L = (np.arange(D) + 1)**0.5
    
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

def Pathological(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( 0.5 + (np.sin((100*X[:, :-1]**2+X[:, 1:]**2)**0.5)**2 - 0.5)/(1+0.001*(X[:, :-1]**2-2*X[:, :-1]*X[:, 1:]+X[:, 1:]**2))**2, axis=1 )
    
    return F

def Paviani(X):
    # X in [2, 10], D fixed 10
    # X* = [9.35027, 9.35027, ..., 9.35027]
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

def Schaffer_F1(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + (np.sin((X1**2+X2**2)**2)**2-0.5)/(1+0.001*(X1**2+X2**2))**2
    
    return F









def Schaffer_F7(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    si = ( (X[:, :-1]**2 + X[:, 1:]**2)**0.5 ).flatten()
    F = 1/(D-1) * si**0.5 * (np.sin(50*si**0.2))**2
    
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

def Schwefel220(X):
    # Schwefel 2.20
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(np.abs(X), axis=1 )
    
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



def Sphere(X):
    # Spherical Contours, Square Sum, Harmonic, De_Jong1 or Schumer-Steiglitz1
    # X in [-100, 100]
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
    # X in [-100, 100]
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
    # X in [-10, 10]
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
    # X in [-20, 20]
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
    # X in [-5, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    L = np.arange(D)
    
    F = np.sum(X**2, axis=1) + np.sum(L/2*X, axis=1)**2 + np.sum(L/2*X, axis=1)**4
    
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