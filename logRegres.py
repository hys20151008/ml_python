# -*- coding: utf-8 -*-

import numpy as np
from math import exp
from scipy.special import expit

def loadDataSet():
    data = np.genfromtxt('./dataset/testSet5.txt')
    dataMat = data[:,0:2]
    labels = data[:, -1]
    return dataMat, labels


def loadDataSet2():
    dataMat = []; labelMat = []
    fr = open('./dataset/testSet5.txt')
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

def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = dataMatrix.shape
    weights = np.ones(n)
    dataIndex = range(m)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
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


def classifyVec(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    return 0


def colicTest():
    frTrain = open('dataset/horseColicTraining.txt')
    frTest = open('dataset/horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount, numTestVec = 0, 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVec(np.array(lineArr), trainWeights))!=int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
 

def multiTest():
    numTests, errorSum = 10, 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

