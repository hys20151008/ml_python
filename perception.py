# -*- coding: utf-8 -*-

import numpy as np

class Perception:
    def __init__(self, learningrate, ndim=2):
        self.lr = learningrate
       # self.wh = np.random.normal(0, 0.1,len(train_x[0]))
        self.wh = np.array([0.0, 0.0])
        self.b = 0

    def train(self,inputs, target):
        inputs = np.asarray(inputs)
      
        
        
        if self.check(inputs,target):
            self.wh += self.lr * np.dot(inputs, target)
            self.b += target * self.lr
    
            print(self.wh, self.b)

    def check(self, inputs, target):
        flag = False
        res = 0.0
        res += (np.dot(inputs, self.wh)+self.b)*target
        if res <=0:
            flag = True
        
        return flag
    
        
        
       




if __name__ == '__main__':
    perception = Perception(1.0)
    inputs = [[3, 3], [4, 3], [1, 1]]
    target = [1, 1, -1]
    while True:    
        test = np.array([])
        for i in range(len(inputs)):
            perception.train(inputs[i],target[i])
        for j in range(len(inputs)):
            if perception.check(inputs[j], target[j]):
                test = np.append(test,1)
                break
        if not test.any():
            break
    print('great')
    
         
    
      
