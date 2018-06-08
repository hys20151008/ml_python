# -*- coding: utf-8 -*-

import numpy as np
from math import exp
from scipy.special import expit

def loadDataSet():
    data = np.genfromtxt('testSet.txt')
    dataMat = data[:,0:2]
    labels = data[:, -1]
    return dataMat, labels


def loadDataSet2():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


# alse can use scipy.special.expit(x)
def sigmoid(inX):
   # return 1.0/(1 + exp(-inX))
    return expit(inX)


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = dataMatrix.shape

    alpha = 0.001
    max_iter = 500
    weights = np.ones((n, 1))
    for k in range(max_iter):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat -h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat,labelMat=loadDataSet2()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
