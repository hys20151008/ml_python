# -*- coding: utf-8 -*-

import os
import numpy as np
import operator
from scipy.spatial import minkowski_distance
from sklearn.preprocessing import normalize

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
#    dataSetSize = dataSet.shape[0]
#    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
#    sqDiffMat = diffMat **2
#    sqDistances = sqDiffMat.sum(axis=1)
#    distances = sqDistances**0.5
#    print('distances', distances)
    distances = minkowski_distance(inX, dataSet)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)

    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

def file2matrix2(filename):
    data = np.genfromtxt(filename, dtype='|U8')
    returnMat = data[:, 0:-1].astype('float64')
    return returnMat, data[:,-1]


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def autoNorm2(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = normalize(dataSet, norm='max',axis=0) 
    return normDataSet, maxVals-minVals, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix2('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm2(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    accuracyCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],1)
        if classfierResult == datingLabels[i]:
            accuracyCount += 1
    accuracy = accuracyCount/numTestVecs
    print("the total accuracy rate is: %f" % accuracy)


def img2vec(filename):
    rVec = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            rVec[0, 32*i+j] = int(line[j])
    return rVec


def handwritingClassTest():
    hwLabels = []
    trainFileList = os.listdir('digits/trainingDigits/')
    m = len(trainFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filename = trainFileList[i]
        className = int(filename.split('_')[0])
        hwLabels.append(className)
        trainingMat[i,:] = img2vec('digits/trainingDigits/%s' % filename)
    testFileList = os.listdir('digits/testDigits')
    accuracyCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filename = testFileList[i]
        className = int(filename.split('_')[0])
        vecTest = img2vec('digits/testDigits/%s' % filename)
        classifierResult = classify0(vecTest, trainingMat, hwLabels, 1)
        if classifierResult == className:
            accuracyCount += 1
    print("the total accuracy rate is: %f" % (accuracyCount/mTest))
