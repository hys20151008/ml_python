# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance


def loadDataSet(filename):
    data = np.genfromtxt(filename)
    dataMat = data[:, 0:]
    return dataMat


def distEclud(x, y):
    return np.sqrt(sum(np.power(x - y, 2)))


def distEclud2(x, y):
    return distance.euclidean(x, y)


def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ, maxJ = np.min(dataSet[:,j]), np.max(dataSet[:,j])
        rangeJ = maxJ - minJ
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kmeans(dataSet, k, distMeas=distEclud2, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud2):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0),dataSet[j,:])**2
        
    while(len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kmeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:,1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit:", sseSplit, sseNotSplit)
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
    bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] ==len(centList)
    bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
    print("the bestCentToSplit is:", bestCentToSplit)
    print("the len of bestClustAss is: ", len(bestClustAss))
    centList[bestCentToSplit] = bestNewCents[0,:]
    centList.append(bestNewCents[1,:])
    clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return np.mat(centList), clusterAssment


    
