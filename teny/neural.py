# -*- coding:utf-8 -*-

import numpy as np
from numpy import random as nrd
import copy

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
        if args and args['clip']:
            self.clip = args['clip']
        self.b = np.random.randn(1, self.out_dim)
        self.w_optimizer, self.b_optimizer = None, None

    def set_index(self, idx):
        '''
        当前层在网络中第几层 从0开始
        '''
        self.idx = idx
    
    def is_activation(self):
        return False

    def set_input_dim(self, input_dim):
        '''
        设置输入纬度
        input_dim: int or list of int
        在 dense 中是 一个 int
        '''
        self.in_dim = input_dim
        self.w = np.random.randn(self.in_dim, self.out_dim) - 0.5

    def input_dim(self):
        return self.in_dim

    def output_dim(self):
        '''
        输出纬度
        out_dim: int or list of int
        在 dense 中是 int
        '''
        return self.out_dim
        
    def calculate(self, x):
        '''
        算子实现
        '''
        self.x = x
        self.y = np.dot(self.x, self.w) + self.b

    def output(self):
        '''
        输出
        '''
        return self.y

    def derivate(self, error):
        '''
        计算误差
        '''
        self.ew = np.dot(self.x.T, error)
        self.eb = error
        self.add_error([self.ew, self.eb])
        return self.error_for_last_layer(error)

    def clear_error_weight(self):
        '''
        初始化/清空误差
        在每个batch前调用
        '''
        a, b = self.input_dim(), self.output_dim()
        self.error_weight = np.zeros([a, b])
        self.error_bias = np.zeros([1, b])
        self.batch_size = 0

    # errors 数组
    def add_error(self, *errors):
        '''
        累计每个样本的误差
        errors: matrix or list of matrix
        在 dense 中，是 weight 和 bias 的误差
        '''
        a, b = errors
        self.error_weight += np.reshape(a, self.error_weight.shape)
        self.error_bias += np.reshape(b, self.error_bias.shape)
        self.batch_size += 1

    def error_for_last_layer(self, der_error):
        '''
        返回对前一层的误差求导
        '''
        if self.idx >0 :
            return np.dot(der_error, self.w.T)
        return None # 第一层 dense 没有前一层，直接返回，不需计算

    def avg_error(self):
        '''
        对批次求平均误差
        '''
        self.error_weight /= float(self.batch_size)
        self.error_bias /= float(self.batch_size)

    def grad(self):
        '''
        对批次的平均误差，进行梯度计算
        '''
        ew = self.w_optimizer.grad(self.error_weight)
        eb = self.b_optimizer.grad(self.error_bias)
        if (self.clip is not None) and (len(self.clip) == 2):
            ew = np.clip(ew, self.clip[0], self.clip[1])
            eb = np.clip(eb, self.clip[0], self.clip[1])
        return ew, eb

    def update(self, error_weight, error_bias):
        '''
        更新参数
        '''
        self.b += error_bias
        self.w += error_weight

    def set_optimizer(self, weight_optimizer_function, bias_optimizer_function):
        self.w_optimizer = weight_optimizer_function
        self.b_optimizer = bias_optimizer_function


# -------------------------------- Convolute --------------------------------
    
# PADDING MODE
PADDING_SAME = 0  # 与输出一致
PADDING_VALID = 1  # 不做对齐补充  会有损失

def __padding(data, fsize, stride, padding=PADDING_SAME):
    '''
    params:
    data: 二维或者三维数据  如果是三维数据，第一个是channel数   channel ,width, height
    fsize: 过滤器大小 二维 [width, height]
    stride: 步长 二维 [width, height]
    padding: PADDING_SAME/PADDING_VALID

    result:
    padding data
    width stride rounds
    height stride rounds
    '''
    if padding == PADDING_VALID:
        return data, int((data.shape[2]-fsize[0])/stride[0])+1, int((data.shape[1]-fsize[1])/stride[1])+1
    elif padding != PADDING_SAME:
        raise Exception('invalid padding in __padding ', padding, ' must be PADDING_SAME(int 0) or PADDING_VALID(int 1)')
    shape = []
    if len(data.shape) == 2:
        shape.append(1)
        shape.extend(data.shape)
    else:
        shape.extend(data.shape[0:3])
    channel, width, height = shape[0], shape[1], shape[2] # 目标纬度
    a = stride[0]*(width-1)+fsize[0]-width
    b = stride[1]*(height-1)+fsize[1]-height
    pad_left, pad_right, pad_up, pad_down = 0, 0, 0, 0
    if a % 2 == 0:
        pad_left = pad_right = int(a / 2)
    else:
        pad_left = int(a // 2)
        pad_right = a - pad_left
    if b % 2 == 0:
        pad_up = pad_down = int(b / 2)
    else:
        pad_up = int(b // 2)
        pad_down = b - pad_up
    if len(data.shape) == 2:
        result = np.zeros([1, pad_left + width + pad_right, pad_up + height + pad_down])
        result[0: 1, pad_left:pad_left+width, pad_up:pad_up+height] = np.reshape(data, [1, data.shape[0], data.shape[1]])
    else:
        result = np.zeros([channel, pad_left + width + pad_right, pad_up + height + pad_down])
        result[0:channel, pad_left:pad_left+width, pad_up:pad_up+height] = data
    return np.array(result), width, height # int((a+width-fsize[0])/stride[0])+1, int((b+height-fsize[1])/stride[1])+1


class Conv:

    def __init__(self, **args):
        '''
        args
        size: filter 大小 二维或者三维数组，如果有三维，第一维是通道数
        padding: same # 与输出一致
                 valid  # 不做对齐补充  会有损失
        stride: filter 移动步长 list of int
        count:  filter 个数 
        '''
        if args and args['size']:
            if len(args['size']) == 3:
                self.channel = args['size'][0]
                self.size = args['size'][1:3]
            else:
                self.channel = 1
                self.size = args['size'][0:2]
        else:
            self.size = [3, 3]  # default
            self.channel = 1
        if args and args['padding']:
            if args['padding'] == 'valid':
                self.padding = 'valid'
            elif args['padding'] == 'same':
                self.padding = 'same'
            else:
                print('invalid padding in convolute neural. default \'same\'')
                self.padding = 'same'
        if args and args['stride']:
            if len(args['stride']) == 1:
                self.cstride = args['stride'][0]  # width
                self.rstride = args['stride'][0]  # height
            else:
                self.cstride = args['stride'][0]
                self.rstride = args['stride'][1]
        else:
            self.cstride = 1
            self.rstride = 1
        if args and args['count']:
            self.count = args['count']
        else:
            self.count = 1
        if args and args['clip']:
            self.clip = args['clip']
        self.init_filters()
        self.init_output_dim()


    def set_index(self, idx):
        self.idx = idx
    
    def index(self):
        return self.idx

    def is_activation(self):
        return False

    def init_filters(self):
        self.filters= []
        self.bias = []
        if self.channel == 1:
            for _ in range(self.count):
                self.filters.append(np.random.normal(size=self.size))
                self.bias.append(np.random.random())
        else:
            s = []
            s.extend(self.size)
            s.append(self.channel)
            for _ in range(self.count):
                self.filters.append(np.random.normal(size=s))
                self.bias.append(np.random.random())
        return
    
    def init_output_dim(self):
        if self.padding == PADDING_SAME:
            a = []
            a.extend(self.in_dim[0:2])
            a.extend(self.count)
            self.out_dim = a
        else:
            w = self.in_dim[0]
            h = self.in_dim[1]
            a = int((w-self.size[0])/self.cstride)+1
            b = int((h-self.size[1])/self.rstride)+1
            self.out_dim = [a, b, self.count]

    def set_input_dim(self, input_dim):
        '''
        input_dim: 二维或者三维，第三维是通道数
        '''
        if len(input_dim) <= 1:
            raise Exception('invalid input_dim in convolute, input_dim: ', input_dim)
        self.in_dim = input_dim
        self.size = input_dim[0:2]
        self.channel = 1
        if len(input_dim) == 3:
            self.channel = input_dim[2]
        self.init_filters()
        self.init_output_dim()

    def input_dim(self):
        return self.in_dim

    def output_dim(self):
        return self.out_dim
        
    # x input
    # 矩阵相乘
    def calculate(self, x):
        self.x = x
        y = self.convolute(x, self.filters, self.bias, self.size, [self.cstride, self.rstride], self.padding)
        self.y = np.array(y).reshape(self.out_dim())

    def convolute(self, x, y, bias, fsize, strides, padding):
        matrix, width_rounds, height_rounds = __padding(x, fsize, strides, padding)
        self.input_matrix = matrix
        self.matrix_width_rounds = width_rounds
        self.matrix_height_rounds = height_rounds
        result = []
        if len(x.shape) == 2:
            for i, f in enumerate(y):
                d = [[self.__convolute(matrix[h*strides[1]:h*strides[1]+fsize[0], w*strides[0]:w*strides[0]+fsize[1]], f, bias[i]) for w in range(width_rounds)] for h in range(height_rounds)]
                result.append(d)
        else:
            for i, f in enumerate(y):
                d = [[self.__convolute(matrix[h*strides[1]:h*strides[1]+fsize[0], w*strides[0]:w*strides[0]+fsize[1], 0:x.shape[2]], f, bias[i]) for w in range(width_rounds)] for h in range(height_rounds)]
                result.append(d)
        return result

    def __convolute(self, m, f, b):
        return np.sum(m*f) + b

    # 返回计算值
    def output(self):
        return self.y

    # 求导 计算误差
    # error
    def derivate(self, error):
        '''
        param:
            error: list(size: count) of errors
        return
            error
        '''
        self.d_filter = [np.zeros_like(self.channel, self.size[0], self.size[1]) for _ in range(self.count)]
        self.d_bias = [0 for i in range(self.count)]
        for i, e in enumerate(error): # count    each filter
            fe = []
            for _ in range(self.channel): # 单个 filter 的 每一个通道
                total = self.size[0] * self.size[1]
                df = []
                for r in range(self.size[0]):
                    for c in range(self.size[1]):
                        _f = np.eye(total)[r*self.size[0] + c].reshape(self.size)
                        df.append(np.sum(e * (self.__convolute(self.input_matrix, _f, 0))))
                fe.append(np.reshape(df, self.size))
            self.d_filter[i] += np.array(fe).reshape(self.channel, self.size[0], self.size[1])
            self.d_bias[i] += np.sum(e)
        self.add_error([self.d_filter, self.d_bias])
        return self.error_for_last_layer(error)

    def clear_error_weight(self):
        '''
        初始化/清空误差
        在每个batch前调用
        '''
        raise Exception('no implementation')
        a, b = self.input_dim(), self.output_dim()
        self.error_weight = np.zeros([a, b])
        self.error_bias = np.zeros([1, b])
        self.batch_size = 0

    # errors 数组
    def add_error(self, *errors):
        '''
        累计每个样本的误差
        errors: matrix or list of matrix
        '''
        raise Exception('no implementation')
        a, b = errors
        # self.error_weight += np.reshape(a, self.error_weight.shape)
        # self.error_bias += np.reshape(b, self.error_bias.shape)
        self.batch_size += 1

    def error_for_last_layer(self, der_error):
        '''
        返回对前一层的误差求导
        '''
        raise Exception('no implementation')
        if self.idx >0 :
            pass
        return None # 第一层 dense 没有前一层，直接返回，不需计算

    def avg_error(self):
        for i in range(self.count):
            self.error_weight[i] /= float(self.batch_size)
            self.error_bias[i] /= float(self.batch_size)

    def grad(self):
        ews, ebs = [], []
        for i in range(self.count):
            ew = self.w_optimizer[i].grad(self.error_weight[i])
            eb = self.b_optimizer[i].grad(self.error_bias[i])
            if (self.clip is not None) and len(self.clip) == 2:
                ew = np.clip(ew, self.clip[0], self.clip[1])
                eb = np.clip(eb, self.clip[0], self.clip[1])
            ews.append(ew)
            ebs.append(eb)
        return ew, eb
        

    def update(self, error_weight, error_bias):
        for i in range(self.count):
            self.filters[i] += error_weight[i]
            self.bias[i] += error_bias[i]

    def set_optimizer(self, weight_optimizer_function, bias_optimizer_function):
        self.w_optimizer, self.b_optimizer = [], []
        for _ in range(self.count):
            self.w_optimizer.append(copy.deepcopy(weight_optimizer_function))
            self.b_optimizer.append(copy.deepcopy(bias_optimizer_function))




# POOLING MODE
POOLING_MAX = 0  # 最大
POOLING_AVG = 1  # 平均
POOLING_ECHO = 2  # 原样返回


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

    def set_index(self, idx):
        self.idx = idx
    
    def index(self):
        return self.idx

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

    def set_index(self, idx):
        self.idx = idx

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

    def set_index(self, idx):
        self.idx = idx

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

    def set_index(self, idx):
        self.idx = idx

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
    
    def set_index(self, idx):
        self.idx = idx
    
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


# -----------------------------------------------------------------------------------------

def test_padding():
    data = np.array([[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 10]])
    fsize = [2, 2]
    strides = [2, 2]
    print(data)
    print(data.shape)
    print('-----------')
    r, a, b = __padding(data, fsize, strides, PADDING_SAME)
    print(np.shape(r))
    print(r)
    print(a, b)

print('np.eye(5)[1]: ', np.eye(5)[1])
a = np.array([[1, 2], [2, 3], [3, 4]])
b = [np.zeros([1, 3, 2]) for _ in range(2)]
b[0] += a
b[1] += a*2
b = np.reshape(b, [2, 1, 3, 2])
print(b.shape)
print(b)

print('---------------------------------')
test_padding()
print('---------------------')
print(np.random.random())