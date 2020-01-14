#  -*- coding:utf-8 -*-

import numpy as np

class Adagrad:

    def __init__(self, learning_rate=0.01, epsilon=0.00001):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.prev_n = None

    def __init_n(self, dim):
        self.prev_n = np.zeros(dim)

    def grad(self, grad):
        if self.prev_n is None:
            self.__init_n(grad.shape)
        self.prev_n = self.prev_n + grad*grad
        return -self.learning_rate*grad/(np.sqrt(self.prev_n+self.epsilon))