# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np

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
    # V3 = random_key(X, N).astype(int)
    V3 = X.astype(int)
    F = np.zeros([P])
    
    for i in range(P):
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        Idle_len = [[] for i in range(M)]
        Idle_st = [[] for i in range(M)]
        Idle_ed = [[] for i in range(M)]
        
        # V3[i] = np.array([0, 1, 0, 3, 3, 2, 3, 5, 4, 0, 4, 3, 4, 5, 5, 1,
        # 1, 2, 1, 5, 5, 3, 0, 0, 2, 1, 4, 4, 2, 2, 1, 4, 3, 2, 0, 5])
        # V3[i] = np.array([3, 3, 1, 5, 2, 3, 4, 5, 0, 0, 4, 4, 0, 3, 5, 0, 2,
        # 0, 4, 2, 1, 1, 5, 5, 2, 0, 2, 3, 1, 5, 2, 4, 1, 1, 4, 3])
        # V3[i] = np.array([0, 4, 5, 2, 3, 1, 3, 4, 1, 4, 2, 3, 0, 2, 5, 5,
        # 4, 3, 0, 1, 4, 5, 3, 5, 2, 0, 1, 4, 2, 2, 0, 5, 1, 0, 3, 1])
        # V3[i] = np.array([1, 3, 3, 1, 4, 5, 4, 3, 5, 2, 5, 3, 1, 2, 0, 2,
        # 0, 2, 0, 1, 2, 4, 3, 1, 1, 4, 0, 4, 5, 5, 2, 0, 4, 0, 3, 5])
        # V3[i] = np.array([ 1, 2, 9,  4,  6,  6, 7, 6,  6,  4,  7,
        #                   10, 8, 4,  6,  6,  7, 5,  9, 10,  5,
        #                    5, 2, 4,  8,  3,  7, 5,  4,  4,  9,
        #                    6, 3, 8, 10, 10,  1, 5,  2,  9,  7,
        #                   10, 9, 5,  7,  7,  4, 6,  8,  2,  6,
        #                    3, 1, 3,  1,  5,  9, 9,  6,  8,  2,
        #                    1, 8, 5,  7, 10, 10, 1,  2,  9,  5,
        #                   10, 9, 4,  3,  7,  1, 3,  3,  8,  8,
        #                   10, 2, 2,  1,  9,  4, 1,  8,  3,  7,
        #                   8,  3, 3,  2,  2,  4, 5, 10,  1])-1  # FT10
        
        for job in V3[i]:
            # 初始化
            need_to_fixed = True
            
            # 取得加工次數operation、機台編號sequence、加工耗時cost
            operation = Operation[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # if operation==1 and sequence==0 and job==0: # 測試用
            #     print(123)
            
            if Idle_len[sequence]: # 若機台編號sequence有閒置時間Idle
                if any(Idle_len[sequence]>=cost): # 若機台編號sequence的閒置時間Idle_len 大於等於 當前的加工耗時cost 為存在
                    condition1 = Idle_len[sequence]>=cost # 判斷每一Idle的長度是否滿足cost
                    condition2 = Idle_ed[sequence]>=Job[job]+cost # 判斷每一Idle的結束時間是否大於Job[job]+cost
                    if operation==0: # 若當前的加工次數為0，則僅需考慮條件1
                        mask = condition1
                    else: # 若當前的加工次數不為0，則需考慮條件1及條件2
                        mask = condition1*condition2
                        
                    if any(mask)==True: # 若遮罩存在True，代表至少有一個Idle可以被使用
                        idx = np.where(mask==True)[0]
                        if len(idx)>1: # 若有多個Idle可以被使用，則默認選擇第1個(接近時刻0)
                            idx = idx[0]
                        idx = int(idx)
                        
                        aaa = np.arange(int(Idle_len[sequence][idx])) + Idle_st[sequence][idx] + 1 # 建立臨時刻度
                        bbb = np.zeros(int(Idle_len[sequence][idx])) # 建立臨時陣列
                        if operation==0: # 若工件job未被加工，則直接從最左開始
                            bbb[:cost] = 1
                        elif aaa.min()>Job[job]: # 若被選定的Idle區間與前一段job完全無交集，則直接從最左開始
                            bbb[:cost] = 1
                        else: # 若被選定的Idle區間與前一段job部分交集
                            skip = aaa.min() - 1
                            bbb[int(Job[job]-skip):int(Job[job]-skip+cost)] = 1
                        
                        Job[job] = aaa[np.where(bbb==1)[0][-1]].copy() # 更新工件時間
                        
                        # 更新Idle
                        Idle_len[sequence].pop(idx)
                        Idle_st[sequence].pop(idx)
                        Idle_ed[sequence].pop(idx)
                        sw = 'st'
                        for t in range(len(bbb)):
                            if bbb[t]==0 and sw=='st':
                                Idle_st[sequence].append(aaa[t]-1)
                                sw = 'ed'
                            
                            if bbb[t]==1 and sw=='ed':
                                Idle_ed[sequence].append(aaa[t-1])
                                Idle_len[sequence].append(Idle_ed[sequence][-1]-Idle_st[sequence][-1])
                                sw = 'st'
                        if sw=='ed':
                            Idle_ed[sequence].append(aaa[-1])
                            Idle_len[sequence].append(Idle_ed[sequence][-1]-Idle_st[sequence][-1])
                            
                        # Job[job] += cost # 僅須更新Job
                        need_to_fixed = False # 不需要修正機台時間Machine及工件時間Job
                    else: # 若遮罩不存在True，代表沒有Idle可以被使用
                        Machine[sequence] += cost
                        Job[job] += cost
                else: # 若機台編號sequence的閒置時間Idle_len 大於等於 當前的加工耗時cost 為不存在
                    Machine[sequence] += cost
                    Job[job] += cost
            else: # 若機台編號sequence沒有閒置時間Idle
                Machine[sequence] += cost
                Job[job] += cost
            
            # 更新加工次數
            Operation[job] += 1
            
            # 新增機台編號sequence的閒置時間Idle
            if Job[job]>Machine[sequence]: # 只有在Job[job]>Machine[sequence]才會有新的閒置時間Idle
                Idle_new = Job[job] - Machine[sequence]
                Idle_len[sequence].append(Idle_new)
                Idle_st[sequence].append(Machine[sequence]-cost)
                Idle_ed[sequence].append(Job[job]-cost)
            
            # 修正機台時間Machine及工件時間Job
            if need_to_fixed==True:
                fixed_time = np.maximum(Machine[sequence], Job[job])
                Machine[sequence] = fixed_time
                Job[job] = fixed_time
            
            # 4. 更新甘特圖
            # print(sequence+1)
            # print(job+1)
            # print(cost)
            # print(Machine)
            # print(Job)
            # print(Operation)
            # print(Idle_len)
            # print(Idle_st)
            # print(Idle_ed)
            # print('-'*30)
            # print()
        
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
