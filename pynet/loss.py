# -*- coding:utf-8 -*

import numpy as np

class BinaryCrossEntropy:
    '''
    loss = -ylog(py) - (1-y)log(1-py)
    '''

    def __init__(self, epsilon=0.000001):
        self.epsilon = epsilon
        self.s = 0.0
        self.count = 0

    # y: real value
    # py: predict value
    def calculate(self, y, py):
        self.s += -y*np.log(py+self.epsilon) - (1-y)*np.log(1-py+self.epsilon)
        self.__devirate(y, py)
        self.count += 1
        
    def loss(self):
        return self.s/self.count

    def error(self):
        return self.e

    def __devirate(self, y, py):
        self.e = -y/(py+self.epsilon) - (1-y)/(1-py+self.epsilon)