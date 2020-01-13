# -*- coding:utf-8 -*-

import numpy as np

'''
--------------------------------------------------- dense layer --------------------------------------------------- 
'''

class Dense:

    # args: input_dim
    def __init__(self, output_dim, **args):
        self.out_dim = output_dim
        self.in_dim = None
        if args and args['input_dim']:
            self.in_dim = args['input_dim']
            self.w = np.random.randn(self.in_dim, self.out_dim)
        self.b = np.random.randn(1, self.out_dim)
        self.w_optimizer, self.b_optimizer = None, None

    def is_activation(self):
        return False

    def set_input_dim(self, input_dim):
        self.in_dim = input_dim
        self.w = np.random.randn(self.in_dim, self.out_dim)

    def input_dim(self):
        return self.in_dim

    def output_dim(self):
        return self.out_dim

    def weight(self):
        return self.w
        
    # x input
    # 矩阵相乘
    def calculate(self, x):
        self.x = x
        self.y = np.dot(self.x, self.w) + self.b

    # 返回计算值
    def output(self):
        return self.y

    # 求导 计算误差
    # error
    def derivate(self, error):
        self.ew = np.dot(self.x.T, error)
        self.eb = error
        return self.ew, self.eb

    def update(self, error_weight, error_bias):
        self.b += error_bias
        self.w += error_weight

    def set_optimizer(self, weight_optimizer_function, bias_optimizer_function):
        self.w_optimizer = weight_optimizer_function
        self.b_optimizer = bias_optimizer_function

    def weight_optimizer(self):
        return self.w_optimizer
    
    def bias_optimizer(self):
        return self.b_optimizer

'''
--------------------------------------------------- activation function --------------------------------------------------- 
'''

# -------------------------------- Relu --------------------------------

class Relu:

    def __init__(self, a):
        a = max(0, a)
        self.a = a

    def is_activation(self):
        return True

    def calculate(self, x):
        self.x = x
        self.y = np.maximum(self.x, self.a * self.x)

    def output(self):
        return self.y

    def derivate(self, e):
        self.ex = np.where(self.y>0, e, self.a*e)
        return self.ex



# -------------------------------- Sigmoid --------------------------------

class Sigmoid:

    def __init__(self):
        pass

    def is_activation(self):
        return True

    def calculate(self, x):
        self.x = x
        self.y = 1.0/(1.0+np.exp(-x))

    def output(self):
        return self.y

    def derivate(self, e):
        self.ex = e * (self.y * (1-self.y))
        return self.ex



# -------------------------------- Tanh --------------------------------

class Tanh:

    def __init__(self):
        pass

    def is_activation(self):
        return True

    def calculate(self, x):
        self.x = x
        self.y = np.tanh(x)
    
    def output(self):
        return self.y

    def derivate(self, e):
        self.ex = e * (1 - self.y * self.y)
        return self.ex

# -------------------------------- dropout --------------------------------

class Dropout:

    # drop 丢弃概率
    def __init__(self, drop):
        self.drop = drop
        self.retain_prob = 1. - drop

    def is_activation(self):
        return True

    def calculate(self, x):
        self.x = x
        self.r = np.random.binomial(n=1, p=self.retain_prob, size=self.x.shape)  #  r = 0|1
        self.y = self.x * self.r / self.retain_prob

    def output(self):
        return self.y

    def derivate(self, e):
        self.ex = e * (self.r / self.retain_prob)
        return self.ex

