# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

# =============================================================================
# main ref
# [1] https://www.al-roomi.org/benchmarks
# [2] http://infinity77.net/global_optimization/genindex.html
# [3] https://opytimark.readthedocs.io/en/latest/api/opytimark.markers.html
# [4] https://www.sfu.ca/~ssurjano/optimization.html
# [5] A Literature Survey of Benchmark Functions For Global Optimization Problems
# =============================================================================

import numpy as np

#%%
# =============================================================================
# 2-D
# =============================================================================
def Adjiman(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [5, 0]
    # F* = -5
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.cos(X1)*np.sin(X2) - X1/(X2**2+1)
    
    return F

def AluffiPentini(X):
    # [1]
    # Zirilli's Function
    # X in [-10, 10], D fixed 2
    # X* = [-1.046680576580755, 0]
    # F* = -0.352386073800034
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.25*X1**4 -0.5*X1**2 + 0.1*X1 + 0.5*X2**2
    
    return F

def BananaShape(X):
    # [1]
    # X in [-1.5, 1.5], D fixed 2
    # X* = [0, 0]
    # F* = -25
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 10*( (X1+1)**2 - (X2+1)**2 ) + X1**2 + 4
    F = -100/F
    
    return F

def BartelsConn(X):
    # [1]
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
    # [1]
    # X in [-4.5, 4.5], D fixed 2
    # X* = [3, 0.5]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1.5-X1+X1*X2)**2 + (2.25-X1+X1*X2**2)**2 + (2.625-X1+X1*X2**3)**2
    
    return F

def BiggsEXP2(X):
    # [1]
    # X in [0, 20], D fixed 2
    # X* = [1, 10]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros(P)
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(9) + 1
    
    for i in range(P):
        F[i] = np.sum( (np.exp(-L*X1[i]/10) -5*np.exp(-L*X2[i]/10) - np.exp(-L/10) + 5*np.exp(-L))**2 )
    
    return F

def Bird(X):
    # [1]
    # X in [-2PI, PI], D fixed 2
    # X* = [4.701055751981055, 3.152946019601391], [-1.582142172055011,-3.130246799635430]
    # F* = -106.7645367198034
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1-X2)**2 + np.sin(X1)*np.e**((1-np.cos(X2))**2) + np.cos(X2)*np.e**((1-np.sin(X1))**2)
    
    return F

def Bohachevsky1(X):
    # [1]
    # X in [-50, 50], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1) - 0.4*np.cos(4*np.pi*X2) + 0.7
    
    return F

def Bohachevsky2(X):
    # [1]
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
    # [1]
    # X in [-50, 50], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + 2*X2**2 - 0.3*np.cos(3*np.pi*X1+4*np.pi*X2) + 0.3
    
    return F

def Booth(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [1, 3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+2*X2-7)**2 + (2*X1+X2-5)**2
    
    return F

def Branin_N1(X):
    # [1]
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

def Branin_N2(X):
    # [1]
    # X1 in [-5, 10], X2 in [0, 15], D fixed 2
    # X* = [-3.196988423389338, 12.526257883092258]
    # F* = -0.179891239069905
    if X.ndim==1:
        X = X.reshape(1, -1)
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    d = 6
    e = 10
    g = 1/(8*np.pi)
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = a * ( X2-b*X1**2+c*X1-d )**2
    f2 = e * (1-g) * np.cos(X1) *np.cos(X2)
    f3  = np.log( X1**2+X2**2+1 )
    
    F = f1 + f2 + f3 + e
    F = -1/F
    
    return F

def Brent(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [-10, -10]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+10)**2 + (X2+10)**2 + np.exp(-X1**2-X2**2)
    
    return F

def Bukin_N4(X):
    # [1]
    # X1 in [-15, -5], X2 in [-3, 3], D fixed 2
    # X* = [-10, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*X2**2 + 0.01*np.abs(X1+10)
    
    return F

def Bukin_N6(X):
    # [1]
    # X1 in [-15, -5], X2 in [-3, 3], D fixed 2
    # X* = [-10, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*np.sqrt(np.abs(X2-0.01*X1**2)) + 0.01*np.abs(X1+10)
    
    return F

def Camel(X):
    # [1]
    # X in [-2, 2], D fixed 2
    # X* = [-1.5, 0], [1.5, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -(-X1**4+4.5*X1**2+2)/np.exp(2*X2**2)
    
    return F

def CarromTable(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [±9.646157266348881, ±9.646157266348881]
    # F* = -24.15681551650653
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1/30 * np.exp(2*np.abs(1-(X1**2+X2**2)**0.5/np.pi)) *np.cos(X1)**2 * np.cos(X2)**2
    
    return F

def ChenBird(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0.5, 0.5]
    # F* = -2000.003999984000
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    b = 0.001
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = b**2 + (X1**2+X2**2-1)**2
    f2 = b**2 + (X1**2+X2**2-1/2)**2
    f3 = b**2 + (X1-X2)**2
    
    F = b/f1 + b/f2 + b/f3
    
    return -F

def ChenV(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0.388888888888889, 0.722222222222222]
    # F* = -2000
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    b = 0.001
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = b**2 + (X1-0.4*X2-0.1)**2
    f2 = b**2 + (2*X1+X2-1.5)**2

    F = b/f1 + b/f2
    
    return -F

def Chichinadze(X):
    # [1]
    # X in [-30, 30], D fixed 2
    # X* = [6.189866586965680, 0.5]
    # F* = -42.94438701899098
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    F = X1**2 - 12*X1 + 11 + 10*np.cos(0.5*np.pi*X1) + 8*np.sin(2.5*np.pi*X1) - 0.2*5**0.5/np.exp(0.5*(X2-0.5)**2)
    
    return F

def Complex(X):
    # [1]
    # X in [-2, 2], D fixed 2
    # X* = [1, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    F = (X1**3-3*X1*X2**2-1)**2 + (3*X2*X1**2-X2**3)**2
    
    return F

def CrossInTray(X):
    # [1]
    # X in [-15, 15], D fixed 2
    # X* = [±1.349406608602084, ±1.349406608602084]
    # F* = -2.062611870822739
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1E-4*( np.abs( np.sin(X1)*np.sin(X2)*np.exp( np.abs(100-(X1**2+X2**2)**0.5/np.pi) ) ) + 1 )**0.1
    
    return F

def CrossLegTable(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1/( np.abs( np.sin(X1)*np.sin(X2)*np.exp( np.abs(100-(X1**2+X2**2)**0.5/np.pi) ) ) + 1 )**0.1
    
    return F

def CrownedCross(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = 0.0001
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 1E-4*( np.abs( np.sin(X1)*np.sin(X2)*np.exp( np.abs(100-(X1**2+X2**2)**0.5/np.pi) ) ) + 1 )**0.1
    
    return F

def Cube(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*(X2-X1**3)**2 + (1-X1)**2
    
    return F

def Damavandi(X):
    # [1], http://cilib.sourceforge.net/apidocs/net/sourceforge/cilib/functions/continuous/unconstrained/Damavandi.html
    # X in [0, 14], D fixed 2
    # X* = [2, 2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    # 當X=[2, 2]，會導致F=0/0=np.nan，故忽略
    mask1 = X1==2
    mask2 = X2==2
    mask3 = mask1 * mask2
    mask4 = ~mask3
    f1 = 1 - np.abs( np.sin(np.pi*(X1[mask4]-2))*np.sin(np.pi*(X2[mask4]-2))/(np.pi**2*(X1[mask4]-2)*(X2[mask4]-2)) )**5
    f2 = 2 + (X1[mask4]-7)**2 + 2*(X2[mask4]-7)**2
    F[mask4] = f1 * f2
    
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

def DekkerAarts(X):
    # [1]
    # X in [-20, 20], D fixed 2
    # X* = [0, ±14.95]
    # F* = -24777
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 1E5*X1**2 + X2**2 - (X1**2+X2**2)**2 + 1E-5*(X1**2+X2**2)**4
    
    return F

def DownhillStep(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = 9
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.floor(10*(10-np.exp(-X1**2-3*X2**2)))/10
    
    return F

def DropWave(X):
    # [1]
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
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [pi, pi]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -np.cos(X1)*np.cos(X2)*np.exp(-(X1-np.pi)**2-(X2-np.pi)**2)
    
    return F

def EggCrate(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + X2**2 + 25*(np.sin(X1)**2+np.sin(X2)**2)
    
    return F

def ElAttarVidyasagarDutta(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [3.4091868222, -2.1714330361]
    # F* = 1.712780354862198
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1**2+X2-10)**2 + (X1+X2**2-7)**2 + (X1**2+X2**3-1)**2
    
    return F

def Engvall(X):
    # [1]
    # X in [-2000, 2000], D fixed 2
    # X* = [1, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**4 + X2**4 + 2*X1**2*X2**2 - 4*X1 + 3
    
    return F

def F26(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.25*X1**4 - 0.5*X1**2 + 0.1*X1 + 0.5*X2**2
    
    return F

def FreudensteinRoth(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [5, 4]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = -13 + X1 + ((5-X2)*X2-2)*X2
    f2 = -29 + X1 + ((X2+1)*X2-14)*X2
    F = f1**2 + f2**2
    
    return F

def Giunta(X):
    # [1]
    # X in [-1, 1], D fixed 2
    # X* = [0.4673200277395354, 0.4673200169591304]
    # F* = 0.06447042053690566
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = 0.6 + np.sum( np.sin(1-X*16/15)**2 - 0.02*np.sin(4-X*64/15) - np.sin(1-X*16/15) , axis=1 )
    
    return F

def GoldsteinPrice(X):
    # [1]
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

def GramacyLee_N2(X):
    # [1]
    # X in [-1.5, 1.5], D fixed 2
    # X* = [-0.707106776321847, -3.324529260087811*1E-9] = [-0.707106776321847, 0]
    # F* = -0.428881942480353
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1 * np.exp(-X1**2-X2**2)
    
    return F

def GramacyLee_N3(X):
    # [1]
    # X in [-1.5, 1.5], D fixed 2
    # X* = [-1.040825908416920, -1.040825908416920]
    # F* = -1.126871745786324
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = -np.prod( np.exp(-(X-1)**2) + np.exp(-0.8*(X+1)**2) - 0.05*np.sin(8*X+0.8), axis=1)
    
    return F

def H1(X):
    # [1]
    # X in [-25, 25], D fixed 2
    # X* = [36PI/13, 28PI/13]
    # F* = 2
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin(X1-X2/8)**2 + np.sin(X2+X1/8)**2
    f2 = (X1-36*np.pi/13)**2 + (X2-28*np.pi/13)**2 + 1
    F = f1 / f2**0.5
    
    return F

def Himmelblau(X):
    # [1]
    # X in [-6, 6], D fixed 2
    # X* = [3, 2], [3.584428340330,-1.848126526964], [-2.805118086953,3.131312518250], [-3.779310253378, -3.283185991286]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1+X2**2-7)**2 + (X1**2+X2-11)**2
    
    return F

def HolderTable(X):
    # [1], [4]
    # X in [-10, 10], D fixed 2
    # X* = [±8.05502, ±9.664590028909654]
    # F* = -19.20850256788675
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1 * np.abs( np.sin(X1)*np.cos(X2) * np.exp( np.abs( 1 - (X1**2+X2**2)**0.5/np.pi ) ) )
    
    return F

def Hosaki(X):
    # [1]
    # X in [0, 10], D fixed 2
    # X* = [4, 2]
    # F* = -2.345811576101292
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1-8*X1+7*X1**2-7/3*X1**3+0.25*X1**4) * X2**2*np.exp(-X2)
    
    return F

def JennrichSampson(X):
    # [1]
    # X in [-1, 11], D fixed 2
    # X* = [0.25782521321500883, 0.25782521381356827]
    # F* = 124.36218235561473896
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[1]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    L = np.arange(10) + 1
    
    for i in range(P):
        F[i] = np.sum( ( 2 + 2*L - (np.exp(L*X1[i])+np.exp(L*X2[i])) )**2 )
    
    return F

def Judge(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0.864787285816574, 1.235748499036571]
    # F* = 16.081730132960381
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[1]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    A = np.array([4.284, 4.149, 3.877, 0.533, 2.211,
                  2.389, 2.145, 3.231, 1.998, 1.379,
                  2.106, 1.428, 1.011, 2.179, 2.858,
                  1.388, 1.651, 1.593, 1.046, 2.152])
    B = np.array([0.286, 0.973, 0.384, 0.276, 0.973,
                  0.543, 0.957, 0.948, 0.543, 0.797,
                  0.936, 0.889, 0.006, 0.828, 0.399,
                  0.617, 0.939, 0.784, 0.072, 0.889])
    C = np.array([0.645, 0.585, 0.310, 0.058, 0.455,
                  0.779, 0.259, 0.202, 0.028, 0.099,
                  0.142, 0.296, 0.175, 0.180, 0.842,
                  0.039, 0.103, 0.620, 0.158, 0.704])
    
    for i in range(P):
        F[i] = np.sum( ( X1[i] + B*X2[i] + C*X2[i]**2 - A )**2 )
    
    return F

def Keane(X):
    # [1]
    # X in [0, 10], D fixed 2
    # X* = [1.393249070031784, 0], [0, 1.393249070031784]
    # F* = -0.673667521146855
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -1*np.sin(X1-X2)**2*np.sin(X1+X2)**2/(X1**2+X2**2)**0.5
    
    return F

def Kearfott(X):
    # [1]
    # X in [-3, 4], D fixed 2
    # X* = [±1.5**0.5, ±0.5**0.5] = [±1.224744871391589, ±0.7071067811865475]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1**2+X2**2-2)**2 + (X1**2-X2**2-1)**2
    
    return F

def Leon(X):
    # [1]
    # X in [-1.2, 1.2], D fixed 2
    # X* = [1, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*(X2-X1**3)**2 + (1-X1)**2
    
    return F

def Matyas(X):
    # [1]
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
    # [1]
    # X1 in [-1.5, 4], X2 in [-3, 3], D fixed 2
    # X* = [-0.5471975602214493, -1.547197559268372]
    # F* = -1.913222954981037
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = - 1.5*X1 + 2.5*X2 + (X1-X2)**2 + np.sin(X1+X2) + 1
    
    return F

def Mineshaft_N3(X):
    # [1]
    # X in [-2, 2], D fixed 2
    # X* = [0.8, 1.3]
    # F* = -7
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = -5 * np.exp(-1000*(X1-0.5)**2-1000*(X2-0.3)**2) - 7 * np.exp(-2000*(X1-0.8)**2-2000*(X2-1.3)**2)
    
    return F

def Mishra_N3(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [-8.466613775046579, -9.998521308999999]
    # F* = -0.184651333342989
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = np.abs( np.cos( np.abs(X1**2+X2)**0.5 ) )**0.5 + (X1+X2)/100
    
    return F

def Mishra_N4(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [-9.941127263635860, -9.999571661999983]
    # F* = -0.199406970088833
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]

    F = np.abs( np.sin( np.abs(X1**2+X2)**0.5 ) )**0.5 + (X1+X2)/100
    
    return F

def Mishra_N5(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [-1.986820662153768, -10]
    # F* = -1.019829519930943
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( ( np.cos(X1)+np.cos(X2) )**2 )**2
    f2 = np.cos( ( np.sin(X1)+np.sin(X2) )**2 )**2
    F = (f1+f2+X1)**2 + 0.01*X1 + 0.1*X2
    
    return F

def Mishra_N6(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [2.886307215440481, 1.823260331422321]
    # F* = -2.283949838474759
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( ( np.cos(X1)+np.cos(X2) )**2 )**2
    f2 = np.cos( ( np.sin(X1)+np.sin(X2) )**2 )**2
    f3 = 0.1*( (X1-1)**2 + (X2-1)**2 )
    F = -np.log((f1-f2+X1)**2) + f3
    
    return F

def Mishra_N8(X):
    # Decanomial Function
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [2, -3]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = X1**10 - 20*X1**9 + 180*X1**8 - 960*X1**7 + 3360*X1**6 - 8064*X1**5 + 13340*X1**4 - 15360*X1**3 + 11520*X1**2 - 5120*X1 + 2624
    f2 = X2**4 + 12*X2**3 + 54*X2**2 +108*X2 + 81
    F = 0.001*( np.abs(f1)+np.abs(f2) )**2
    
    return F

def Mishra_N10a(X):
    # SeqP Function No.01
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0], [2, 2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = X1 + X2
    f2 = X2 * X2
    F = ( f1 - f2 )**2
    
    return F

def Mishra_N10b(X):
    # SeqP Function No.02
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0], [2, 2]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = X1 + X2
    f2 = X2 * X2
    F = np.abs( f1 - f2 )
    
    return F

def ModifiedSchaffer_N1(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + ( np.sin( X1**2+X2**2 )**2 - 0.5 ) / ( 1 + 0.001*( X1**2+X2**2 ) )**2
    
    return F

def ModifiedSchaffer_N2(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( X1**2-X2**2 )**2 - 0.5
    f2 = ( 1 + 0.001*( X1**2+X2**2 ) )**2
    F = 0.5 + f1/f2
    
    return F

def ModifiedSchaffer_N3(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [±1.253114962205510, 0] or [0, ±1.253114962205510]
    # F* = 0.001566854526004
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.sin( np.cos( np.abs(X1**2-X2**2) ) )**2 - 0.5
    f2 = ( 1 + 0.001*(X1**2+X2**2) )**2
    F = 0.5 + f1/f2
    
    return F

def ModifiedSchaffer_N4(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [±1.253131828792882, 0] or [0, ±1.253131828792882]
    # F* = 0.292578632035980
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.cos( np.sin( np.abs(X1**2-X2**2) ) )**2 - 0.5
    f2 = ( 1 + 0.001*(X1**2+X2**2) )**2
    F = 0.5 + f1/f2
    
    return F

def MullerBrown(X):
    # [1]
    # X1 in [-1.5, 1.5], X2 in [-0.5, 2.5], D fixed 2
    # X* = [-0.558223638251928, 1.441725828290487]
    # F* = -146.6995172099539
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    X1 = X[:, 0]
    X2 = X[:, 1]
    A = np.array([-200, -100, -170, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x1j = np.array([1.0, 0.0, -0.5, -1.0])
    x2j = np.array([0.0, 0.5, 1.5, 1.0])
    
    for i in range(P):
        F[i] = np.sum( A * np.exp( a*(X1[i]-x1j)**2 + b*(X1[i]-x1j)*(X2[i]-x2j) + c*(X2[i]-x2j)**2 ) )
    
    return F

def Parsopoulos(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [±1.57079633, ±0], [±4.71238898, ±0], [±1.57079633, ±3.14159265], [±4.71238898, ±3.14159265]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]

    F = np.cos(X1)**2 + np.sin(X2)**2
    
    return F

def Peaks(X):
    # [1]
    # X in [-4, 4], D fixed 2
    # X* = [0.228279999979237, -1.625531071954464]
    # F* = -6.551133332622496
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = 3*(1-X1)**2 * np.exp(-X1**2-(X2+1)**2)
    f2 = 10*(X1/5-X1**3-X2**5) * np.exp(-X1**2-X2**2)
    f3 = np.exp(-(X1+1)**2-X2**2) / 3
    F = f1 - f2 - f3
    
    return F

def PenHolder(X):
    # [1]
    # X in [-11, 11], D fixed 2
    # X* = [±9.646167671043401, ±9.646167671043401]
    # F* = -0.9635348327265058
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -np.exp(-1/np.abs(np.cos(X1)*np.cos(X2)*np.exp(np.abs(1-np.sqrt(X1**2+X2**2)/np.pi))))
    
    return F

def PowellBadlyScaled(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [1.098*1E-5, 9.106] != [0, 9.106]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (1E4*X1*X2-1)**2 + (np.exp(-X1)+np.exp(-X2)-1.0001)**2
    
    return F

def Price_N1(X):
    # Becker-Lago's Function
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [±5, ±5]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (np.abs(X1)-5)**2 + (np.abs(X2)-5)**2
    
    return F

def Price_N2(X):
    # Periodic Function
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0, 0]
    # F* = 0.9
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 1 + np.sin(X1)**2 + np.sin(X2)**2 - 0.1*np.exp(-X1**2-X2**2)
    
    return F

def Price_N3(X):
    # Modified Rosenbrock's or Price-Rosenbrock's Function
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [1, 1], [0.341307503353524, 0.116490811845416]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 100*(X2-X1**2)**2 + (6.4*(X2-0.5)**2-X1-0.6)**2
    
    return F

def Price_N4(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0, 0], [2, 4], [1.464352119663698, -2.506012760781662]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (2*X1**3*X2-X2**3)**2 + (6*X1-X2**2+X2)**2
    
    return F

def Quadratic(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [0.193880169366971, 0.485133920218833]
    # F* = -3873.724182186271819
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -3803.84 - 138.08*X1 - 232.92*X2 + 128.08*X1**2 + 203.64*X2**2 + 182.23*X1*X2
    
    return F

def RotatedEllipse_N1(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 7*X1**2 - 6*3**0.5*X1*X2 + 13*X2**2
    
    return F

def RotatedEllipse_N2(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 - X1*X2 + X2**2
    
    return F

def Rump(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.abs( (333.75-X1**2)*X2**6 + X1**2*(11*X1**2*X2**2-121*X2**4-2) + 5.5*X2**8 + X1/(2+X2) )
    
    return F

def S2(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [any, 0.7]
    # F* = 2
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 2 + (X2-0.7)**2
    
    return F

def S3(X):
    # [1]
    # X in [-10, 10], D fixed 2
    # X* = [10, 0.7]
    # F* = 0.528872325696265
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 2 + (X2-0.7)**2 - np.arctan(X1)
    
    return F

def Sawtoothxy(X):
    # [1]
    # X in [-20, 20], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    r = (X1**2+X2**2)**0.5
    t = np.arctan2(X2, X1)
    f1 = (np.sin(r) - np.sin(2*r)/2 + np.sin(3*r)/3 - np.sin(4*r)/4 + 4) * (r**2/(r+1))
    f2 = 0.5 * np.cos(2*t-0.5) + np.cos(t) + 2
    F = f1 * f2
    
    return F

def Schaffer_N6(X):
    # [1]
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

def Schaffer_N7(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = (X1**2+X2**2)**0.25
    f2 = 50*(X1**2+X2**2)**0.1 + 1
    F = f1 * f2
    
    return F

def ShekelFoxholes(X):
    # De Jong 5
    # [1]
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

def Six_Hump_Camel_Back(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [-0.08984201368301331, 0.7126564032704135], [0.08984201368301331, -0.7126564032704135]
    # F* = -1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + X1**6/3 + X1*X2 - 4*X2**2 + 4*X2**4
    
    return F

def Stenger(X):
    # [1]
    # X in [-1, 4], D fixed 2
    # X* = [0, 0], [1.695415196279268, 0.718608171943623]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1**2-4*X2)**2 + (X2**2-2*X1+4*X2)**2
    
    return F

def Storn(X, m=1):
    # [1]
    # X(m=1) in [-2, 2], X(m=2) in [-4, 4], X(m=3) in [-8, 8], D fixed 2
    # X*(m=1) = [0, ±1.386952327146511], X*(m=2) = [0, ±2.608906424592038], X*(m=3) = [0, ±4.701739810796703]
    # F*(m=1) = -0.407461605632581, F*(m=2) = -18.058696657349238, F*(m=3) = -227.7657499670953
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 10**m*X1**2 + X2**2 - (X1**2+X2**2)**2 + 10**-m*(X1**2+X2**2)**4
    
    return F

def TestTubeHolder(X):
    # [1]
    # X in [-1, 4], D fixed 2
    # X* = [-PI/2, 0]
    # F* = -10.872299901558
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    f1 = np.exp( np.abs( np.cos( (X1**2+X2**2)/200 ) ) )
    f2 = np.abs( np.sin(X1)*np.cos(X2)*f1 )
    F = -4 * f2
    
    return F

def ThreeCylinders(X):
    # [1]
    # X in [0, 5], D fixed 2
    # X* = ?
    # F* = -1.05
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    F = np.zeros([P])
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    r1 = ( (X1-3)**2 + (X2-2)**2 )**0.5
    r2 = ( (X1-4)**2 + (X2-4)**2 )**0.5
    r3 = ( (X1-1)**2 + (X2-3)**2 )**0.5
    
    mask1 = r1>=0.75
    mask2 = r2>=0.375
    mask3 = r3>=0.375
    
    F[mask1] = 1
    F[mask2] = 1.05
    F[mask3] = 1.05
    
    return -F

def ThreeHumpCamelBack(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 2*X1 - 1.05*X2**4 + X1**6/6 + X1*X2 + X2**2
    
    return F

def Treccani(X):
    # [1]
    # X in [-5, 5], D fixed 2
    # X* = [0, 0], [-2, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**4 + 4*X1**3 + 4*X1**2 + X2**2
    
    return F

def Trefethen(X):
    # [1]
    # X1 in [-6.5, 6.5], X2 in [-4.5, 4.5], D fixed 2
    # X* = [-0.02440307923, 0.2106124261]
    # F* = -3.3068686474
    if X.ndim==1:
        X = X.reshape(1, -1)

    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = np.exp(np.sin(50*X1)) + np.sin(60*np.exp(X2)) + np.sin(70*np.sin(X1)) + np.sin(np.sin(80*X2)) - np.sin(10*(X1+X2)) + 0.25*(X1**2+X2**2)
    
    return F

def Tripod(X):
    # [1]
    # X in [-100, 100], D fixed 2
    # X* = [0, -50]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    p1 = ( X[:, 0]>=0 )*1
    p2 = ( X[:, 1]>=0 )*1
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = p2*(1+p1) + np.abs( X1+50*p2*(1-2*p1) ) + np.abs( X2+50*(1-2*p2) )
    
    return F

def Tsoulos(X):
    # [1]
    # X in [-1, 1], D fixed 2
    # X* = [0, 0]
    # F* = -2
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 + X2**2 - np.cos(18*X1) - np.cos(18*X2)
    
    return F

def Ursem_N1(X):
    # [1]
    # X1 in [-2.5, 3], X2 in [-2, 2], D fixed 2
    # X* = [1.697136443570341, 0]
    # F* = -4.816814063734823
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -np.sin(2*X1-0.5*np.pi) - 3*np.cos(X2) - 0.5*X1
    
    return F

def Ursem_N3(X):
    # [1]
    # X1 in [-2, 2], X2 in [-1.5, 1.5], D fixed 2
    # X* = [0, 0]
    # F* = -2.5
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -(3-np.abs(X1))/2 * (2-np.abs(X2))/2 * np.sin(2.2*np.pi*X1+0.5*np.pi) - (2-np.abs(X1))/2 * (2-np.abs(X2))/2 * np.sin(0.5*np.pi*X2**2+0.5*np.pi)
    
    return F

def Ursem_N4(X):
    # [1]
    # X in [-2, 2], D fixed 2
    # X* = [0, 0]
    # F* = -1.5
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -3*np.sin(0.5*np.pi*X1+0.5*np.pi) * (2-(X1**2+X2**2)**0.5)/4
    
    return F

def UrsemWaves(X):
    # [1]
    # X1 in [-0.9, 1.2], X2 in [-1.2, 1.2], D fixed 2
    # X* = [-0.605689494589848, -1.177561933039789]
    # F* = -7.306998731324462
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = -(0.3*X1)**3 + (X2**2-4.5*X2**2)*X1*X2 + 4.7*np.cos( 3*X1-X2**2*(2+X1) ) * np.sin(2.5*np.pi*X1)
    
    return F

def VenterandSobiezcczanskiSobieski(X):
    # [1]
    # X in [-50, 10], D fixed 2
    # X* = [0, 0]
    # F* = 1000
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = X1**2 - 100*np.cos(X1)**2 - 100*np.cos(X1**2/30) + X2**2 - 100*np.cos(X2)**2 - 100*np.cos(X2**2/30) + 1400
    
    return F

def WayburnSeader_N1(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [1, 2], [1.596804153876933, 0.806391692246134]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = (X1**6+X2**4-17)**2 + (2*X1+X2-4)**2
    
    return F

def WayburnSeader_N2(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [0.200138974728779, 1], [0.424861025271221, 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = ( 1.613 - 4*(X1-0.3125)**2 - 4*(X2-1.625)**2 )**2 + (X2-1)**2
    
    return F

def WayburnSeader_N3(X):
    # [1]
    # X in [-500, 500], D fixed 2
    # X* = [5.146896745324582, 6.839589743000071]
    # F* = 19.105879794567979
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 2*X1**3/3 - 8*X1**2 + 33*X1 - X1*X2 + 5 + ( (X1-4)**2 + (X2-5)**2 - 4 )**2
    
    return F

def XinSheYang_N4(X, K=10):
    pass

def Zettl(X):
    # [1]
    # X in [-1, 5], D fixed 2
    # X* = [-0.02989597760285287, 0]
    # F* = -0.003791237220468656
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.25*X1 + ( X1**2 - 2*X1 + X2**2 )**2
    
    return F