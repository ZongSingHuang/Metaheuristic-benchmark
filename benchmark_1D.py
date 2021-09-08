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
# 1-D
# =============================================================================
def BirdLike(X):
    # [1]
    # X in [-4, 4], D fixed 1
    # X* = [0]
    # F* = 2
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = (2*X1**4 + X1**2 + 2)/(X1**4 + 1)
    
    return F

def Discontinuous(X):
    # [1]
    # X in [0, 10], D fixed 1
    # X* = [6]
    # F* = -1.159146731333981
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    mask1 = X1<6
    mask2 = ~mask1
    
    F[mask1] = X1[mask1]/3 * np.sin(X1[mask1]) * np.cos(X1[mask1])
    F[mask2] = X1[mask2]/4 * np.sin(0.9*X1[mask2])
    
    return F

def DixonSzego(X):
    # [1]
    # X in [-5, 5], D fixed 1
    # X* = [0], [2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = 4*X1**2 - 4*X1**3 + X1**4
    
    return F

def Dixon(X):
    # [1]
    # X in [-5, 5], D fixed 1
    # X* = [-1]
    # F* = -7.5
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = X1**4 - 3*X1**3 - 1.5*X1**2 + 10*X1
    
    return F

def Dolan_N01(X):
    # [1]
    # X in [-100, 100], D fixed 1
    # X* = [7.810207524564704]
    # F* = -703.7287810900712
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = X1**4 - 12*X**3 + 15*X**2 + 56*X1 -60
    
    return F

def GramacyLee_N01(X):
    # [1]
    # X in [0.5, 2.5], D fixed 1
    # X* = [0.548563444114526]
    # F* = -0.869011134989500
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = np.sin(10*np.pi*X1)/(2*X1) + (X1-1)**4
    
    return F

def Himmelblau_N01(X):
    # [1]
    # X in [-2, 2], D fixed 1
    # X* = [-1], [1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = (X1**2 - 1)**2
    
    return F

def Himmelblau_N03(X):
    # [1]
    # X in [-2, 2], D fixed 1
    # X* = [0.409951714917356]
    # F* = -2.267543938143762
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = (1-X1)**2 * (X1+1)**4 * (X1-2)**3 * X1
    
    return F

def InfiniteLimits(X):
    # [1]
    # X in [-100, 100], D fixed 1
    # X* = [-4.493409471849579], [4.493409471849579]
    # F* = -0.217233628211222
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = np.sin(X1)/X1
    
    return F

def KiselevaStepanchuk(X):
    # [1]
    # X in [-2, 2], D fixed 1
    # X* in [0, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = np.abs(X1) + np.abs(X1-1) - 1
    
    return F

def Mineshaft_N01(X):
    # [1]
    # X in [0, 10], D fixed 1
    # X* = [5]
    # F* = 1.380487165157852
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = np.cos(X1) + np.abs((7-X1)**(2/15)) + 2*np.abs((5-X1)**(4/35))
    
    return F

def Mineshaft_N02(X):
    # [1]
    # X in [-10, 10], D fixed 1
    # X* = [2.000454648]
    # F* = -1.416353520337699
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    F = np.cos(X1) - np.exp(-1000*(X1-2)**2)
    
    return F

def Moore(X):
    # [1]
    # 50th Degree Polynomial Problem
    # X in [1, 5], D fixed 1
    # X* = [1.091165036224843]
    # F* = -663.5000966105010
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    L = np.arange(50) + 1
    a = np.array([-500, 2.5, 1.666666666, 1.25, 1.0, 
                  0.8333333, 0.714285714, 0.625, 0.555555555, 1.0,
                  -43.6363636, 0.41666666, 0.384615384, 0.357142857, 0.3333333,
                  0.3125, 0.294117647, 0.277777777, 0.263157894, 0.25,
                  0.238095238, 0.227272727, 0.217391304, 0.208333333, 0.2,
                  0.192307692, 0.185185185, 0.178571428, 0.344827586, 0.6666666,
                  -15.48387097, 0.15625, 0.1515151, 0.14705882, 0.14285712,
                  0.138888888, 0.135135135, 0.131578947, 0.128205128, 0.125,
                  0.121951219, 0.119047619, 0.116279069, 0.113636363, 0.1111111,
                  0.108695652, 0.106382978, 0.208333333, 0.408163265, 0.8])
    X1 = X[:, 0]
    for i in range(P):
        F[i] = np.sum(a*X1[i]**L)
        
    return F

def Problem_N01(X):
    # [1]
    # Rokne's Function No.01
    # X in [-1.5, 11], D fixed 1
    # X* = [10]
    # F* = -2976.323333333334
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = -1/6*(X1**6) + 52/25*(X1**5) - 39/80*(X1**4) - 71/10*(X1**3) + 79/20*(X1**2) + X1 - 1/10
        
    return -F

def Problem_N02(X):
    # [1]
    # Timonov's Function No.01
    # X in [2.7, 7.5], D fixed 1
    # X* = [5.145735284853897]
    # F* = -1.899599349152114
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = -np.sin(X1) - np.sin(10/3*X1)
        
    return -F

def Problem_N03(X):
    # [1]
    # Shubert's, Trigonometric Polynomial or Suharev-Zilinskas' Function
    # X in [-10, 10], D fixed 1
    # X* = [-6.77457614347361], [-0.491390835930674], [5.79179447080188]
    # F* = -12.031249442167146
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    L = np.arange(5) + 1
    F = np.zeros([P])
    X1 = X[:, 0]
    
    for i in range(P):
        F[i] = np.sum( L*np.sin((L+1)*X1[i] + L) )
        
    return -F

def Problem_N04(X):
    # [1]
    # Gaffney's or Cornelius-Lohner's Function
    # X in [1.9, 3.9], D fixed 1
    # X* = [2.868033983115523]
    # F* = -3.850450708800221
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = (16*X1**2-24*X1+5)*np.exp(-X1)
        
    return -F

def Problem_N05(X):
    # [1]
    # Basso's Function
    # X in [0, 1.2], D fixed 1
    # X* = [0.966085802901471]
    # F* = -1.489072538689604
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = (-3*X1+1.4)*np.sin(18*X1)
        
    return -F

def Problem_N06(X):
    # [1]
    # Zilinskas-Shaltyanis' or Richard Brent's Function No.01
    # X in [-10, 10], D fixed 1
    # X* = [0.679578666600993]
    # F* = -0.824239398476077
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = ( X1+np.sin(X1) ) * np.exp(-X1**2)
        
    return -F

def Problem_N07(X):
    # [1]
    # Strongin-Zilinskas-Shaltyanis' or Timonov's Function No.02
    # X in [-2.7, 7.5], D fixed 1
    # X* = [5.199778366858]
    # F* = -1.601307546494396
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = -np.sin(X1) - np.sin(10/3*X1) - np.log(X1) + 0.84*X1 - 3
        
    return -F

def Problem_N08(X):
    # [1]
    # Modified Trigonometric Polynomial or Lévy's Function No.02
    # X in [-10, 10], D fixed 1
    # X* = [-7.08350640682890], [-0.800321099691209], [5.48286420658132]
    # F* = -14.508007927195038
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    L = np.arange(5) + 1
    
    for i in range(P):
        F[i] = np.sum( L*np.cos( (L+1)*X1[i] + L) )
        
    return -F

def Problem_N09(X):
    # [1]
    # Timonov's Function No.03 or Zilinskas' Function No.01
    # X in [3.1, 20.4], D fixed 1
    # X* = [17.039198942112002]
    # F* = -1.905961118715785
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = -np.sin(X1) - np.sin(2/3*X1)
        
    return -F

def Problem_N10(X):
    # [1]
    # Fichtenholz's Function No.01 or Himmelblau's Function No.02
    # X in [-7.85, 7.85], D fixed 1
    # X* = [-4.913180450455435], [4.913180450455435]
    # F* = -4.814469889712269
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1*np.sin(X1)
        
    return -F

def Problem_N11(X):
    # [1]
    # Marsden-Weinstein's Function
    # X in [-PI/2, 2PI], D fixed 1
    # X* = [2.094395090000627], [4.188790191441036]
    # F* = -1.5
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = -2*np.cos(X1) - np.cos(2*X1)
        
    return -F

def Problem_N12(X):
    # [1]
    # Fichtenholz's Function No.02
    # X in [0, 2PI], D fixed 1
    # X* = [PI], [3PI/2]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = -np.sin(X1)**3 - np.cos(X1)**3
        
    return -F

def Problem_N13(X):
    # [1]
    # Fichtenholz's Function No.03
    # X in [0.001, 0.99], D fixed 1
    # X* = 1/1.414213562373095
    # F* = -1.587401051968199
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1**(2/3) + (1-X1**2)**(1/3)
        
    return -F

def Problem_N14(X):
    # [1]
    # Fichtenholz's Function No.04
    # X in [0, 4], D fixed 1
    # X* = [0.224880383897449]
    # F* = -0.788685387408673
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = np.exp(-X1) * np.sin(2*np.pi*X1)
        
    return -F

def Problem_N15(X):
    # [1]
    # Fichtenholz's Function No.05
    # X in [-5, 5], D fixed 1
    # X* = [2.414213560038049]
    # F* = -0.035533905932738
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = (-X1**2+5*X1-6) / (X1**2+1)
        
    return -F

def Problem_N16(X):
    # [1]
    # Phillips' Function
    # X in [-3, 3], D fixed 1
    # X* = [1.590717100915575]
    # F* = 7.515924153082323
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = -2*(X1-3)**2 - np.exp(X1**2/2)
        
    return -F

def Problem_N17(X):
    # [1]
    # Lévy's Function No.01
    # X in [-4, 4], D fixed 1
    # X* = [-3], [3]
    # F* = 7
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = -X1**6 + 15*X1**4 - 27*X1**2 - 250
        
    return -F

def Problem_N18(X):
    # [1]
    # Timonov's Function No.04
    # X in [0, 6], D fixed 1
    # X* = [2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    
    mask1 = X1<=3
    mask2 = ~mask1
    F[mask1] = -(X1[mask1]-2)**2
    F[mask2] = -2*np.log(X1[mask2]-2) - 1
    
    return -F

def Problem_N19(X):
    # [1]
    # Lévy-Gomez's Function
    # X in [0, 6.5], D fixed 1
    # X* = [5.872865514594919]
    # F* = -7.815674542981392
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1 - np.sin(3*X1) + 1
    
    return -F

def Problem_N20(X):
    # [1]
    # Richard Brent's Function No.02
    # X in [-10, 10], D fixed 1
    # X* = [1.195136633593035]
    # F* = -0.063490528936440
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = (X1-np.sin(X1)) * np.exp(-X1**2)
    
    return -F

def Problem_N21(X):
    # [1]
    # X in [0, 10], D fixed 1
    # X* = [4.795408682338769]
    # F* = -9.508350440633096
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1*np.sin(X1) + X1*np.cos(2*X1)
    
    return F

def Problem_N22(X):
    # [1]
    # X in [0, 20], D fixed 1
    # X* = [9PI/2]
    # F* = (exp(-27PI/2)) - 1 = [-1]
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = np.exp(-3*X1) - np.sin(X1)**3
    
    return F

def Problem_N1_01(X):
    # [1]
    # X in [-1, 2], D fixed 1
    # X* = [1.950519411050002]
    # F* = 0.049740265515763
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1*np.sin(10*np.pi*X1) + 2
    
    return F

def Problem_N1_02(X):
    # [1]
    # X in [0, 9], D fixed 1
    # X* = [0.891723942104979]
    # F* = -15.164402119605699
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = X1 + 10*np.sin(5*X1) + 7*np.cos(4*X1)
    
    return F

def Problem_N1_sen(X):
    # [1]
    # X in [0, 1], D fixed 1
    # X* = [0.1], [0.3], [0.5], [0.7], [0.9]
    # F* = 1
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    F = np.sin(5*np.pi*X1)**6
    
    return F

def Pseudoethane(X):
    # [1]
    # X in [0, 2PI], D fixed 1
    # X* = [3.201787176863421]
    # F* = -1.071114593111043
    if X.ndim==1:
        X = X.reshape(1, -1)
    r0 = 1.54
    theta = 109.5*np.pi/180
    X1 = X[:, 0]
    
    f1 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1-2*np.pi/3)-np.cos(theta)**2 )*r0**2
    f2 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1-2*np.pi/3)-np.cos(theta)**2 )*r0**2
    f3 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1)-np.cos(theta)**2 )*r0**2
    f4 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1)-np.cos(theta)**2 )*r0**2
    f5 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1+2*np.pi/3)-np.cos(theta)**2 )*r0**2
    f6 = 3*r0**2 - 4*r0**2*np.cos(theta) - 2*( np.sin(theta)**2*np.cos(X1+2*np.pi/3)-np.cos(theta)**2 )*r0**2
    
    F = 588600/f1**6 - 1079.1/f2**3 + 600800/f3**6 - 1071.5/f4**3 + 481300/f5**6 - 1064.6/f6**3
    
    return F

def Rokne_N2(X):
    # [1]
    # X in [0, 2], D fixed 1
    # X* = [1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = (X1-1)**10
    
    return F

def S1(X):
    # [1]
    # X in [-10, 10], D fixed 1
    # X* = [1], [2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = (X1-1)**2 * (X1-2)**2
    
    return F

def Strongin(X):
    # [1]
    # X in [-1.5, 4.5], D fixed 1
    # X* = [0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = 2 - np.cos(X1) - np.cos(2*X1)
    
    return F

def Suharev(X):
    # [1]
    # X in [0, 1], D fixed 1
    # X* = 1/(3PI/2 + 2PI) = [0.09094568176679733]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = np.sin(1/X1)
    
    return F

def Wilkinson(X):
    # [1]
    # X in [1, 10], D fixed 1
    # X* = [6.325654058491549]
    # F* = -443.6717047411253
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = 0.000089248*X1 - 0.0218343*X1**2 + 0.998266*X1**3 - 1.6995*X1**4 + 0.2*X1**5
    
    return F

def Zilinskas_N2(X):
    # [1]
    # X in [-2, 2], D fixed 1
    # X* = [0.75]
    # F* = -1.125
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = 2*(X1-0.75)**2 + np.sin(8*np.pi*X1 - np.pi/2) - 0.125
    
    return F

def Zilinskas_N3(X):
    # [1]
    # X in [0, 100], D fixed 1
    # X* = 3PI/2 + 2PI = [10.995574287564276]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    X1 = X[:, 0]
    
    F = np.sin(X1)
    
    return F