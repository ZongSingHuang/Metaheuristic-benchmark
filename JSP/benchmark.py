# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:43:29 2021

@author: zongsing.huang
"""

import numpy as np
import pandas as pd
import plotly_express as px
import plotly.io as pio
# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

def fitness(X, M, N, Sequence, Cost):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    X = X.astype(int)
    F = np.zeros([P])
    critical_path = []
    
    for i in range(P):
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        list_start = {}
        list_end = {}
        
        # X[i] = np.array([2, 3, 2, 1, 1, 4, 4, 3, 4, 2, 3, 1, 3, 1, 2, 4], dtype=int) - 1
        
        for idx, job in enumerate(X[i]):
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
            list_start[idx] = fixed_time - cost
            list_end[idx] = fixed_time
            
        # 5. makespan
        F[i] = Machine.max()

        # 6. 關鍵路徑
        critical_path.append(CPM(list_start, list_end, F[i]))
        
    return F, critical_path

def CPM(list_start, list_end, makespan):
    tree = []
    copy_list_end = list_end.copy()
    for idx, val in copy_list_end.items():
        if val==makespan:
            node = {}
            node[int(idx)] = list_start[int(idx)]
            tree.append(node)
            del list_start[int(idx)]
            del list_end[int(idx)]
    del copy_list_end
    
    for branch_idx, branch in enumerate(tree):
        end_of_branch = branch[list(branch)[-1]]
        
        for leaf_idx, leaf_val in list_end.items():
            if leaf_val==end_of_branch:
                node_val = list_start[leaf_idx]
                node_idx = leaf_idx
                copy_branch = branch.copy()
                copy_branch[int(node_idx)] = node_val
                tree.append(copy_branch)
    
    critical_path = []
    for idx, branch in enumerate(tree):
        end_of_branch = branch[list(branch)[-1]]
        if end_of_branch==0:
            critical_path += list(branch)
    critical_path = list(set(critical_path))
    
    return critical_path

def random_key(X, N):
    if X.ndim==1:
        X = X.reshape(1, -1)

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

def convert_to_datetime(x):
  return pd.to_datetime(x, unit='D')

def gantt(X, M, N, Sequence, Cost):
    if X.ndim!=1:
        X = X.reshape(-1)
        
    X = X.astype(int)
    Data = pd.DataFrame(columns=['Job', 'Operation', 'Machine', 'Start', 'Cost', 'End'])
    Machine = np.zeros([M])
    Job = np.zeros([N])
    Operation = np.zeros([N], dtype=int)
    
    for job in X:
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
        
        # 4. 寫入表
        Data.loc[-1] = [job+1, Operation[job], str(sequence+1),
                        convert_to_datetime(fixed_time-cost),
                        convert_to_datetime(cost),
                        convert_to_datetime(fixed_time)]
        Data.index = Data.index + 1
        Data = Data.sort_index()
        
    # 5. 更新甘特圖
    Data = Data.sort_values(by=['Machine'])
    fig = px.timeline(Data, x_start="Start", x_end="End", y="Machine", color='Job')
    num_tick_labels = np.linspace(start = 0, stop = int(Machine.max()), num = int(Machine.max()+1), dtype = int)
    date_ticks = [convert_to_datetime(x) for x in num_tick_labels]
    fig.layout.xaxis.update({'tickvals' : date_ticks,
                             'ticktext' : num_tick_labels})
    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
    
    fig.show()
    return 0

def fitness2(X, M, N, Sequence, Cost):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    X = X.astype(int)
    F = np.zeros([P])
    
    for i in range(P):
        gantt = np.zeros([M, np.sum(Cost)]) - 1
        gantt2 = np.zeros([M, np.sum(Cost)]) - 66666
        Machine = np.zeros([M])
        Job = np.zeros([N])
        Operation = np.zeros([N], dtype=int)
        Idle_len = [[] for i in range(M)]
        Idle_st = [[] for i in range(M)]
        Idle_ed = [[] for i in range(M)]
        
        ct = 0
        for job in X[i]:
            # 初始化
            need_to_fixed = True
            
            # 取得加工次數operation、機台編號sequence、加工耗時cost
            operation = Operation[job]
            sequence = Sequence[job, operation]
            cost = Cost[job, operation]
            
            # if operation==0 and sequence==2 and job==4: # 測試用
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
                        
                        # 更新甘特圖
                        gantt[sequence, int(aaa[np.where(bbb==1)[0][0]]-1):int(aaa[np.where(bbb==1)[0][0]]-1+cost)] = job
                        gantt2[sequence, int(aaa[np.where(bbb==1)[0][0]]-1):int(aaa[np.where(bbb==1)[0][0]]-1+cost)] = X[i, ct]
                        
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
                
                # 更新甘特圖
                gantt[sequence, int(fixed_time-cost):int(fixed_time)] = job
                gantt2[sequence, int(fixed_time-cost):int(fixed_time)] = X[i, ct]
            ct = ct + 1
        
        # 修復
        dect = np.zeros(M) - 1
        V3_fixed = []
        X_fixed = []
        for t in range(gantt.shape[1]):
            if np.array_equal(dect, gantt[:, t])==False:
                for k in range(M):
                    if dect[k]!=gantt[k, t] and gantt[k, t]!=-1:
                        V3_fixed.append(gantt[k, t])
                        X_fixed.append(gantt2[k, t])
                        
                dect = gantt[:, t].copy()
        V3_fixed = np.array(V3_fixed).astype(int)
        X_fixed = np.array(X_fixed)
        X[i] = V3_fixed.copy()
        
        # makespan
        F[i] = Machine.max()
        
    return X, F
