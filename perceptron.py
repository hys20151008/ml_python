# -*- coding: utf-8 -*-

import numpy as np

class Perception:
    def __init__(self, learningrate):
        self.lr = learningrate
        self.wh = np.array([0.0, 0.0])
        self.b = 0

    def train(self,inputs, target):
        inputs = np.asarray(inputs)
      
        # if not correct,update wh and b
        if self.check(inputs,target):
            self.wh += self.lr * np.dot(inputs, target)
            self.b += target * self.lr
    
            print(self.wh, self.b)
    
    # check inputs whether classify correct,if correct return False,else True
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
        # Check that all inputs are correctly classified,if any wrong,break,then return to train
        for j in range(len(inputs)):
            if perception.check(inputs[j], target[j]):
                test = np.append(test,1)
                break
        if not test.any():
            break
    print('great')

