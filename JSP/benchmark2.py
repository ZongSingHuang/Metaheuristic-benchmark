# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np
import time

def ft06(X, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    
    Cost = np.array([[1, 3, 6, 7, 3, 6],
                     [8, 5, 10, 10, 10, 4],
                     [5, 4,  8,  9,  1, 7],
                     [5, 5,  5,  3,  8, 9],
                     [9, 3,  5,  4,  3, 1],
                     [3, 3,  9, 10,  4, 1]])
    Sequence = np.array([[2, 0, 1, 3, 5, 4],
                         [1, 2, 4, 5, 0, 3],
                         [2, 3, 5, 0, 1, 4],
                         [1, 0, 2, 3, 4, 5],
                         [2, 1, 4, 5, 0, 3],
                         [1, 3, 5, 0, 4, 2]], dtype=int)

    F = fitness(P, M, N, Sequence, Cost, X)
        
    return F

def ft10(X, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    
    Cost = np.array([[29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
                     [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
                     [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
                     [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
                     [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
                     [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
                     [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
                     [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
                     [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
                     [85, 13, 61,  7, 64, 76, 47, 52, 90, 45]])
    Sequence = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
                         [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
                         [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
                         [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
                         [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
                         [1, 0, 3, 2, 6, 5, 9, 8, 7, 4], 
                         [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
                         [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
                         [1, 0, 2 ,6, 8, 9, 5, 3, 4, 7]], dtype=int)

    F = fitness(P, M, N, Sequence, Cost, X)
        
    return F

def ft20(X, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    
    Cost = np.array([[29,  9, 49, 62, 44],
                     [43, 75, 69, 46, 72],
                     [91, 39, 90, 12, 45],
                     [81, 71,  9, 85, 22],
                     [14, 22, 26, 21, 72],
                     [84, 52, 48, 47,  6],
                     [46, 61, 32, 32, 30], 
                     [31, 46, 32, 19, 36],
                     [76, 76, 85, 40, 26],
                     [85, 61, 64, 47, 90],
                     [78, 36, 11, 56, 21],
                     [90, 11, 28, 46, 30],
                     [85, 74, 10, 89, 33],
                     [95, 99, 52, 98, 43],
                     [ 6, 61, 69, 49, 53],
                     [ 2, 95, 72, 65, 25],
                     [37, 13, 21, 89, 55],
                     [86, 74, 88, 48, 79],
                     [69, 51, 11, 89, 74],
                     [13, 7, 76, 52, 45]])
    Sequence = np.array([[0, 1, 2, 3, 4],
                         [0, 1, 3, 2, 4],
                         [1, 0, 2, 4, 3],
                         [1, 0, 4, 2, 3],
                         [2, 1, 0, 3, 4],
                         [2, 1, 4, 0, 3],
                         [1, 0, 2, 3, 4], 
                         [2, 1, 0, 3, 4],
                         [0, 3, 2, 1, 4],
                         [1, 2, 0, 3, 4],
                         [1, 3, 0, 4, 2],
                         [2, 0, 1, 3, 4],
                         [0, 2, 1, 3, 4],
                         [2, 0, 1, 3, 4],
                         [0, 1, 4, 2, 3],
                         [1, 0, 3, 4, 2],
                         [0, 2, 1, 3, 4],
                         [0, 1, 4, 2, 3],
                         [1, 2, 0, 3, 4],
                         [0, 1, 2, 3, 4]], dtype=int)

    F = fitness(P, M, N, Sequence, Cost, X)
        
    return F

def la01(X, M, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    
    Cost = np.array([[21, 53, 95, 55, 34],
                     [21, 52, 16, 26, 71],
                     [39, 98, 42, 31, 12],
                     [77, 55, 79, 66, 77],
                     [83, 34, 64, 19, 37],
                     [54, 43, 79, 92, 62],
                     [69, 77, 87, 87, 93],
                     [38, 60, 41, 24, 83],
                     [17, 49, 25, 44, 98],
                     [77, 79, 43, 75, 96]])
    Sequence = np.array([[1, 0, 4, 3, 2],
                         [0, 3, 4, 2, 1],
                         [3, 4, 1, 2, 0],
                         [1, 0, 4, 2, 3],
                         [0, 3, 2, 1, 4],
                         [1, 2, 4, 0, 3],
                         [3, 4, 1, 2, 0],
                         [2, 0, 1, 3, 4],
                         [3, 1, 4, 0, 2],
                         [4, 3, 2, 1, 0]], dtype=int)

    F = fitness(P, M, N, Sequence, Cost, X)
        
    return F

#%%
def fitness(P, M, N, Sequence, Cost, X):
    X = X.astype(int)
    F = np.zeros([P])
    
    for i in range(P):
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        
        for job in X[i]:
            # 1. 取得Job的Operation之工時與機台
            operation = Operation[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # 2. 更新時間與次數
            Machine[sequence] += cost
            Job[job] += cost
            Operation[job] += 1
            
            # 3. 修正時間
            fixed_time = np.maximum(Machine[sequence], Job[job])
            Machine[sequence] = fixed_time
            Job[job] = fixed_time
            
            # 4. 更新甘特圖
        
        # makespan
        F[i] = Machine.max()
        
    return F

def random_key(X, N):
    if X.ndim==1:
        X = X.reshape(1, -1)
    # N = 3
    # X = np.array([[1.3, 0.7, 2.4, 1.1, 3.4, 5.3],
    #               [0.7, 2.4, 1.3, 1.1, 3.4, 5.3],
    #               [3.7, 1.1, 2.3, 4.6, 6.5, 5.1]])
    P = X.shape[0]
    D = X.shape[1]
    V3 = np.zeros_like(X)
    cont = np.zeros([3, D])
    
    for i in range(P):
        cont[0] = np.arange(D)
        cont[1] = X[i].copy()
        
        idx = X[i].argsort()
        cont = cont[:, idx]
        
        cont[2] = np.arange(D) + 1
        
        idx = cont[0].argsort()
        cont = cont[:, idx]
        
        V3[i] = cont[2] % N
    
    V3.astype(int)
    return V3
