# -*- coding:utf-8 -*-

import numpy as np

'''
--------------------------------------------------- dense layer --------------------------------------------------- 
'''

class Dense:

    def __init__(self, output_dim, input_dim):
        if input_dim:
            self.in_dim = input_dim
        self.out_dim = output_dim
        self.weight = np.random.randn(self.in_dim, self.out_dim)
        self.b = np.random.randn(1, self.out_dim)

    def __set_index(self, i):
        self.index = i
    
    def __is_activation(self):
        return False

    def __set_input_dim(self, input_dim):
        self.in_dim = input_dim

    def __input_dim(self):
        return self.in_dim

    def __output_dim(self):
        return self.out_dim

    def __weight(self):
        return self.weight
        
    # x input
    # 矩阵相乘
    def __calculate(self, x):
        self.x = x
        self.y = np.dot(self.x, self.weight) + self.b

    # 返回计算值
    def __output(self):
        return self.y

    # 求导 计算误差
    # error
    def __derivate(self, error):
        self.ew = np.dot(self.x.T, error)
        self.eb = error
        return self.ew, self.eb

    def __update(self, error_weight, error_bias):
        self.b += error_bias
        self.weight += error_weight

'''
--------------------------------------------------- activation function --------------------------------------------------- 
'''

# -------------------------------- Relu --------------------------------

class Relu:

    def __init__(self, a):
        a = max(0, a)
        self.a = a

    def __set_index(self, i):
        self.index = i

    def __is_activation(self):
        return True

    def __calculate(self, x):
        self.x = x
        self.y = np.maximum(self.x, self.a * self.x)

    def __output(self):
        return self.y

    def __derivate(self, e):
        if self.y > 0:
            self.ex = e
        else:
            self.ex = self.a * e
        return self.ex

# -------------------------------- Sigmoid --------------------------------

class Sigmoid:

    def __inti__(self):
        pass

    def __is_activation(self):
        return True

    def __calculate(self, x):
        self.x = x
        self.y = 1/(1+np.exp(-x))

    def __output(self):
        return self.y

    def __derivate(self, e):
        self.ex = e * (self.y * (1-self.y))
        return self.ex

# -------------------------------- Tanh --------------------------------

class Tanh:

    def __init__(self):
        pass

    def __set_index(self, i):
        self.index = i

    def __is_activation(self):
        return True

    def __calculate(self, x):
        self.x = x
        self.y = np.tanh(x)
    
    def __output(self):
        return self.y

    def __derivate(self, e):
        self.ex = e * (1 - self.y * self.y)
        return self.ex

# -------------------------------- dropout --------------------------------

class Dropout:

    # drop 丢弃概率
    def __init__(self, drop):
        self.drop = drop
        self.retain_prob = 1. - drop

    def __set_index(self, i):
        self.index = i

    def __is_activation(self):
        return True

    def __calculate(self, x):
        self.x = x
        self.r = np.random.binomial(n=1, p=self.retain_prob, size=self.x.shape)  #  r = 0|1
        self.y = self.x * self.r / self.retain_prob

    def __output(self):
        return self.y

    def __derivate(self, e):
        self.ex = e * (self.r / self.retain_prob)
        return self.ex