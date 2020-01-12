#  -*- coding:utf-8 -*-

import numpy as np

class Adagrad:

    def __init__(self, init_n, learning_rate=0.08, epsilon=0.00001):
        self.prev_n = init_n
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def grad(self, grad):
        self.prev_n = self.prev_n + grad*grad
        return -self.learning_rate*grad/(np.sqrt(self.prev_n+self.epsilon))