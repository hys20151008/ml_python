# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet(filename):
    data = np.genfromtxt(filename)
    dataMat = data[:, 0:-1]
    labels = data[:, -1]
    return dataMat, labels


def standRegres(x, y):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


def ridgeRegress(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is a singular, can not do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = xMat.shape
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat



