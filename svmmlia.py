# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet(filename):
    data = np.genfromtxt(filename)
    dataMat = data[:,0:2]
    label = data[:,-1]
    return dataMat, label


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random(0,m))
    reutrn j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

