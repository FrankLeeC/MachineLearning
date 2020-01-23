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
            self.w = np.random.randn(self.in_dim, self.out_dim) - 0.5
        self.b = np.random.randn(1, self.out_dim)
        self.w_optimizer, self.b_optimizer = None, None

    def is_activation(self):
        return False

    def set_input_dim(self, input_dim):
        self.in_dim = input_dim
        self.w = np.random.randn(self.in_dim, self.out_dim) - 0.5

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


# -------------------------------- Convolute --------------------------------

class Conv2d:

    def __init__(self, **args):
        '''
        args
        size: filter 大小
        padding: same # 与输出一致
                 valid  # 不做对齐补充  会有损失
        stride: filter 移动步长
        count:  filter 个数 
        channel: filter 通道数

        '''
        if args and args['size']:
            self.size = args['size']
        else:
            self.size = 3  # default
        if args and args['padding']:
            if args['padding'] == 'valid':
                self.padding = 'valid'
            elif args['padding'] == 'same':
                self.padding = 'same'
            else:
                print('invalid padding in convolute neural. default \'same\'')
                self.padding = 'same'

        if args and args['stride']:
            self.stride = int(args['stride'])
        else:
            self.stride = 1
        if args and args['count']:
            self.count = int(args['count'])
        else:
            self.count = 1
        if args and args['channel']:
            self.channel = int(args['channel'])
        else:
            self.channel = 1
        return

    def is_activation(self):
        return False

    def set_input_dim(self, input_dim):
        if len(input_dim) <= 1:
            raise Exception('invalid input_dim in convolute, input_dim: ', input_dim)
        self.in_dim = input_dim

    def input_dim(self):
        return self.size

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

    
# PADDING MODE
PADDING_SAME = 0  # 与输出一致
PADDING_VALID = 1  # 不做对齐补充  会有损失

# POOLING MODE
POOLING_MAX = 0  # 最大
POOLING_AVG = 1  # 平均
POOLING_ECHO = 2  # 原样返回

# 2维卷积
# matrix的层数与filters层数一致
# matrix  单个正方形图层通道
# filters 过滤器集合  可以多个过滤器
# bias 偏移量 集合
# stride filter 移动步长
# padding 对齐方式
# 返回与过滤器个数相同的卷积层
def convolute2d(matrix, filters, bias, stride, padding=PADDING_SAME):
    fsize, _ = filters[0].shape
    matrix, rounds = __padding(matrix, fsize, stride, padding)
    result = []
    for i, f in enumerate(filters):
        f_width, _ = f.shape
        matrix.shape
        d = [[__conv2d(matrix[r*stride:r*stride+f_width,c*stride:c*stride+f_width], f, bias[i]) for c in range(rounds)] for r in range(rounds)]
        result.append(d)
    return np.array(result)

def __conv2d(m, f, b):
    return np.sum(m*f) + b

# matrix 正方形图层通道 集合
# fsize 窗口大小
# bias 偏移量
# round = [(n+2*p-f)//s]+1  round == out_width
# PADDING_SAME    round = out_width = n  p = [s*(n-1)+f-n]/2
# PADDING_VALID       p = 0   round == out_width = [(n-fsize)//stride]+1
def pooling(matrix, fsize, bias, stride, padding=PADDING_SAME, mode=POOLING_MAX):
    result = []
    for _, data in enumerate(matrix):
        data, rounds = __padding(data, fsize, stride, padding)
        f = __max
        if mode == POOLING_AVG:
            f = __avg
        if mode == POOLING_ECHO:
            f = __echo
        result.append([[f(data[r*stride:r*stride+fsize,c*stride:c*stride+fsize])+bias for c in range(rounds)] for r in range(rounds)])
    return np.array(result)

def __max(matrix):
    return np.max(matrix)

def __avg(matrix):
    return np.average(matrix)

def __echo(matrix):
    return matrix

def __padding(data, fsize, stride, padding=PADDING_SAME):
    n, _ = data.shape
    if padding == PADDING_VALID:
        return data, ((n-fsize)//stride)+1
    a = stride*(n-1)+fsize-n
    pad_left, pad_right = 0, 0
    if a % 2 == 0:
        pad_left = pad_right = a / 2
    else:
        pad_left = a // 2
        pad_right = a - pad_left
    result = np.zeros([pad_left + n + pad_right, pad_left + n + pad_right])
    result[pad_left:pad_left+n,pad_left:pad_left+n] = data
    return result, n


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


# -------------------------------- softmax --------------------------------

class Softmax:

    def __init__(self):
        pass

    def is_activation(self):
        return True

    def calculate(self, x):
        m = np.max(x)
        s = np.exp(x-m)
        self.y = s/np.sum(s)

    def output(self):
        return self.y
    
    def derivate(self, e):
        ex = np.ones_like(self.y)
        for i in range(len(self.y[0])):
            a=np.ones_like(self.y) * -self.y[0][i]
            a *= self.y
            a[0][i] += self.y[0][i]
            ex[0][i] *= np.sum(e*a)
        self.ex = ex
        return self.ex