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
    
    F = 100*(X1**2-X2)**2 + (X1-1)**2+ (X3-1)**2 + 90*(X3**2-X4)**2  + 10.1*((X2-1)**2+(X4-1)**2) + 19.8*(X2-1)*(X4-1)
    
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







def Schaffer_F6(X):
    # X in [-100, 100], D fixed 2
    # X* = [0, 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 0.5 + (np.sin((X1**2+X2**2)**0.5)**2-0.5)/(1+0.001*(X1**2+X2**2))**2
    
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

def Six_Hump_Camel_Back(X):
    # X in [-5, 5], D fixed 2
    # X* = [-0.08984201368301331, 0.7126564032704135] or [0.08984201368301331, -0.7126564032704135]
    # F* = -1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + 1/3*X1**6 + X1*X2 - 4*X2**2 + 4*X2**4
    
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

def Zettl(X):
    # X in [-5, 10], D fixed 2
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