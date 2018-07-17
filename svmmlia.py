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
        j = int(np.random.randint(0,m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = dataMatrix.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("J not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[i,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold)*\
                     dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))

            if(alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
    return b, alphas
               

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i,:], kTup)


def clacEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T*\
          (os.X*os.X[k,:].T)) + os.b
    Ek = fXk - float(os.labelMat[k])
    return Ek


def selectJ(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(os.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
    return j, Ej


def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
    Ei = calcEk(os, i)
    if((os.labelMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or\
      ((os.labelMat[i]*Ei > os.tol) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i, os, Ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if(os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * os.K[i,j] - os.K[i,i] - os.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        os.alphas[j] -= os.labelMat[j]*(Ei - Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEk(os, j)
        if (abs(os.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        os.alphas[i] += os.labelMat[j]*os.labelMat[i]*(alphaJold - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i] - os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
        b2 = os.b - Ej- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j]- os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else: 
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    os = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet, alphaPairsChanged = True, 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, os)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return os.b, os.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w


def kernelTrans(X, A, kTup):
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("That Kernel is not recognized")
    return K


def calcEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T*os.K[:,k] + os.b)
    Ek = fXk - float(os.labelMat[k])
    return Ek


def testRbf(k1=1.3):
    data, label = loadDataSet("dataset/testSetRBF.txt")
    b, alphas = smoP(data, label, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(data)
    labelMat = np.mat(label).transpose()

    svInd = np.nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % sVs.shape[0])

    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(label[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

    data, label = loadDataSet("dataset/testSetRBF2.txt")
    errorCount = 0
    dataMat = np.mat(data)
    labelMat = np.mat(label).transpose()
    m, n = dataMat.shape

    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(label[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

