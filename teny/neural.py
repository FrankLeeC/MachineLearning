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
        对批次的平均误差 进行求导
        '''
        ew = self.w_optimizer.grad(self.error_weight)
        eb = self.b_optimizer.grad(self.error_bias)
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

PADDING_SAME = 0
PADDING_VALID = 1
def __padding(data, fsize, stride, padding=PADDING_SAME):
    '''
    data: 二维或者三维数据
    fsize: 过滤器大小 二维
    stride: 步长 二维
    padding: PADDING_SAME/PADDING_VALID
    '''
    if padding == PADDING_VALID:
        return data
    elif padding != PADDING_SAME:
        raise Exception('invalid padding in __padding ', padding, ' must be PADDING_SAME(int 0) or PADDING_VALID(int 1)')
    shape = []
    if len(data.shape) == 2:
        shape.extend(data.shape)
        shape.extend(1)
    else:
        shape.extend(data.shape[0:3])
    width, height, channel = shape[0], shape[1], shape[2]
    a = stride[0]*(width-1)+fsize[0]-width
    b = stride[1]*(height-1)+fsize[1]-height
    pad_left, pad_right, pad_up, pad_down = 0, 0, 0, 0
    if a % 2 == 0:
        pad_left = pad_right = a / 2
    else:
        pad_left = a // 2
        pad_right = a - pad_left
    if b % 2 == 0:
        pad_up, pad_down = b / 2
    else:
        pad_up = b // 2
        pad_down = b - pad_up
    result = np.zeros([pad_left + width + pad_right, pad_up + height + pad_down, channel])
    result[pad_left:pad_left+width,pad_up:pad_up+height,:] = data
    return result


class Conv2d:

    def __init__(self, **args):
        '''
        args
        size: filter 大小 二维或者三维数组，第三维是通道数
        padding: same # 与输出一致
                 valid  # 不做对齐补充  会有损失
        stride: filter 移动步长 list of int
        count:  filter 个数 
        '''
        if args and args['size']:
            self.size = args['size'][0:2]
            if len(args['size']) == 3:
                self.channel = args['size'][2]
        else:
            self.size = [3, 3]  # default
            self.channel = args['size'][2]
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
                self.cstride = args['stride'][0]
                self.rstride = args['stride'][0]
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
        return

    def set_index(self, idx):
        self.idx = idx
    
    def index(self):
        return self.idx

    def is_activation(self):
        return False

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

    def input_dim(self):
        return self.in_dim

    def output_dim(self):
        if self.padding == 'SAME':
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
        return self.out_dim
        
    # x input
    # 矩阵相乘
    def calculate(self, x):
        pass

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