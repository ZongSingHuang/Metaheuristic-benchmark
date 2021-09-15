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