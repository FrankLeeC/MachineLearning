# -*- coding:utf-8 -*

import numpy as np

class BinaryCrossEntropy:
    '''
    loss = -ylog(py) - (1-y)log(1-py)
    '''

    def __init__(self):
        pass

    # y: real value
    # py: predict value
    def calculate(self, y, py):
        self.s = -y*np.log(py) - (1-y)*np.log(1-py)
        self.__devirate(y, py)
        
    def loss(self):
        return self.s

    def error(self):
        return self.e

    def __devirate(self, y, py):
        self.e = -y/py - (1-y)/(1-py)