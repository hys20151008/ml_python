# -*- coding: utf-8 -*-


import numpy  as np
from collections import Counter


def createDataSet():
    data = np.array([[1, 1, -1],[1, 2, -1],[1, 2, 1], [1, 1, 1],[1, 1, -1],
        [2,1,-1],[2,2,-1],[2,2,1],[2,3,1],[2,3,1],[3,3,1],[3,2,1],[3,2,1],
        [3,3,1],[3,3,-1]])
    #return data[:,0:-1],data[:,-1]
    return data


def train(dataset,label):
    n = len(label)
    data = dataset
    p1 = 9/15
    
    p_postive = np.zeros((2,3))
    p_negtive = np.zeros((2,3))

    for i in range(2):
        uni = list(set(dataset[:,i]))
        for j in range(len(uni)):
            
            f = data[data[:,i] == uni[j]]

            q = len(data[data[:,-1] == 1])
            p = len(f[f[:,-1] == 1])
            p_postive[i][j] = p/q

            pp = len(f[f[:,-1] == -1])
            qq = len(data[data[:,-1] == -1])
            p_negtive[i][j] = pp/qq
    return p_postive, p_negtive, p1


def predit(feature,p_postive, p_negtive, p1):
    l = [0,0]
    l[0] = p1 * p_postive[0][feature[0]-1] * p_postive[1][feature[-1]-1]
    l[-1] = (1-p1) * p_negtive[0][feature[0]-1] * p_negtive[1][feature[-1]-1]
    if l[0] > l[-1]:
        y = 1
    else:
        y = -1
    return y
         
             

