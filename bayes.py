# -*- coding: utf-8 -*-

import numpy as np
from math import log


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
        'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', \
        'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', \
        'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
        'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# unique the dataset
def createVecabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


# convert words to vec, if inputset in vacabList,relative index value is set 1, else 0
def setOfWords2Vec(vocabList, inputSet):
    retVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return retVec


def bagOfWords2VecMN(vocabList, inputSet):
    retVec = [0] * len(vocabList)
    for w in inputSet:
        if w in vocabList:
            retVec[vocabList.index(w)] +=1
    return retVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    print(numTrainDocs)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            print(trainMatrix[i])
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    print(p1Denom)
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive
    

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVecabList(listPosts)
    trainMat = []
    for p in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, p))

    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        wordList = textParse(open('dataset/email/spam/%d.txt' % i, 'r', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('dataset/email/ham/%d.txt' % i, 'r', errors='ignore').read())
        print(i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVecabList(docList)
    trainingSet = range(50)
    testSet = []

    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
#        del(trainingSet[randIndex])

    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount)/len(testSet))
 
        
