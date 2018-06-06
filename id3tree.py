# -*- coding: utf-8 -*-


import operator
import numpy as np
from math import log
from collections import Counter
from scipy.stats import entropy


def calcEnt(dataSet):
    num = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCount:
            labelCount[label] = 0
        labelCount[label] += 1

    Ent = 0.0
    for k in labelCount:
        prob = float(labelCount[k])/num
        Ent -= prob * log(prob, 2)
    return Ent


def calcEnt2(dataSet):
    dataSet = np.array(dataSet)
    num = len(dataSet)
    labelCount = Counter(dataSet[:,-1])
    
    prod = [float(labelCount[k])/num for k in labelCount]
    Ent = entropy(prod, base=2)

    return Ent
    
    
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitData(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    hd = calcEnt2(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    
    for i in range(numFeatures):
        featList = [elt[i] for elt in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for v in uniqueVals:
            subDataSet = splitData(dataSet, i, v)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcEnt2(subDataSet)
        infoGain = hd - newEntropy
        print(infoGain)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
            

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [elt[-1] for elt in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        print('dataSet', dataSet)
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [elt[bestFeat] for elt in dataSet]
    uniqueVals = set(featValues)
    print('labels=', labels)

    for v in uniqueVals:
        subLabels = labels[:]
        print('subLabels',subLabels)
        myTree[bestFeatLabel][v] = createTree(splitData(dataSet, bestFeat, v), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    print('featIndex', featIndex)
    print('secondDict', secondDict)
    for k in secondDict:
        if testVec[featIndex] == k:
            if isinstance(secondDict[k], dict):
                classLabel = classify(secondDict[k], featLabels, testVec)
            else:
                classLabel = secondDict[k]
    return classLabel 
