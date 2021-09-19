# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

import numpy as np

def Sphere():
    return [-100, 100]

def Rastrigin():
    return [-5.12, 5.12]

def Ackley():
    return [-32, 32]

def Griewank():
    return [-600, 600]

def Schwefel_P222():
    return [-10, 10]

def Rosenbrock():
    return [-30, 30]

def Sehwwefel_P221():
    return [-100, 100]

def Quartic():
    return [-1.28, 1.28]

def Schwefel_P12():
    return [-100, 100]

def Penalized1():
    return [-50, 50]

def Penalized2():
    return [-50, 50]

def Schwefel_226():
    return [-500, 500]

def Step():
    return [-100, 100]

def Kowalik():
    return [-5, 5]

def ShekelFoxholes():
    return [-65.536, 65.536]

def GoldsteinPrice():
    return [-2, 2]

def Shekel():
    return [0, 10]

def Branin():
    # X1 in [-5, 10], X2 in [0, 15], D fixed 2
    return [-5, 0, 10, 15]

def Hartmann3():
    return [0, 1]

def SixHumpCamelBack():
    return [-5, 5]

def Hartmann6():
    return [0, 1]

def Zakharov():
    return [-5, 10]

def SumSquares():
    return [-10, 10]

def Alpine():
    return [-10, 10]

def Michalewicz():
    return [0, np.pi]

def Exponential():
    return [-1, 1]

def Schaffer():
    return [-100, 100]

def BentCigar():
    return [-100, 100]

def Bohachevsky1():
    return [-50, 50]

def Elliptic():
    return [-100, 100]

def DropWave():
    return [-5.12, 5.12]

def CosineMixture():
    return [-1, 1]

def Ellipsoidal(D):
    return [-D, D]

def LevyandMontalvo1():
    return [-10, 10]

def Easom():
    return [-10, 10]

def SumofDifferentPower():
    return [-1, 1]

def LevyandMontalvo2():
    return [-5, 5]

def Holzman():
    return [-10, 10]

def XinSheYang1():
    return [-20, 20]

def XinSheYang6():
    return [-10, 10]

def Beale():
    return [-4.5, 4.5]

def Shubert():
    return [-10, 10]

def InvertedCosineMixture():
    return [-1, 1]

def Salomon():
    return [-100, 100]

def Matyas():
    return [-10, 10]

def Leon():
    return [-1.2, 1.2]

def Paviani():
    return [2.001, 9.999]

def Sinusoidal():
    return [0, np.pi]

def ktablet():
    return [-5.12, 5.12]

def NoncontinuousRastrigin():
    return [-5.12, 5.12]

def Fletcher():
    return [-np.pi, np.pi]

def Levy():
    return [-10, 10]

def Davis():
    return [-100, 100]

def Pathological():
    return [-100, 100]

def Schwefel_P220():
    return [-100, 100]

def Booth():
    return [-10, 10]

def Zettl():
    return [-1, 5]

def PowellQuartic():
    return [-1, 5]

def Tablet():
    return [-1, 1]

def Brown():
    return [-1, 4]

def ChungReynolds():
    return [-100, 100]

def Csendes():
    return [-1, 1]

def Bohachevsky2():
    return [-50, 50]

def Bohachevsky3():
    return [-50, 50]

def Colville():
    return [-10, 10]

def BartelsConn():
    return [-500, 500]

def Bird():
    return [-2*np.pi, 2*np.pi]