# -*- coding: utf-8 -*-

import numpy as np

def makeGram(inputs):
    m = len(inputs)
    gramatrix = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            gramatrix[i][j] = np.dot(inputs[i], inputs[j])
    return gramatrix


class PerceptionPair:
    def __init__(self, dataSet, target, learningrate=1):
        self.lr = learningrate
        self.a = np.zeros(len(dataSet), np.float)
        self.b = 0.0
        self.gram = np.matmul(np.array(dataSet), np.array(dataSet).T)
        self.target = target


    def calc(self,i):
        res = np.dot(self.a*self.target, self.gram[i])
        res = (res+self.b)*target[i]
        return res

    def train(self, inputs, target):
        flag = False
        m = len(inputs)
        res = 0.0
        for i in range(m):
            if self.calc(i) <= 0:
                self.a[i] += self.lr
                self.b += target[i]
                print(self.a, self.b)


  


if __name__ == '__main__':
    inputs = [[3, 3], [4, 3], [1, 1]]
    target = [1, 1, -1]
    perception = PerceptionPair(dataSet=inputs,target=target)
     
    while True:
        test = np.array([])
        for i in range(len(inputs)):
            perception.train(inputs, target)
        for j in range(len(inputs)):
            if perception.calc(j) <= 0:
                test = np.append(test, 1)
                break
        if not test.any():
            w = np.dot(perception.a*perception.target, inputs)
            print('w: ',end='')
            print(w)
            print('b: ',end='')
            print(perception.b)
            break
    print("great")
        

        
