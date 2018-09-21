# -*-encode=utf-8-*-
import codecs

import numpy as np
from scipy.misc import imread
import os
import datetime

theta, x, y, px, py = [], [], [], [], []
k = 10
alpha = 0.001
m = 60000
tm = 10000
times = 25
weight_lambda = 0.002


def read(path):
    _x, _y = [], []
    base = path
    for root, dirs, files in os.walk(base):
        for f in files:
            path = os.path.join(root, f)
            label = path[base.__len__() + 1: base.__len__() + 2]
            _y.append([int(label)])
            image = imread(path)
            image = image.reshape(image.shape[0] * image.shape[1], 1) * 0.01
            _x.append(image.transpose().tolist()[0])
    _x = np.mat(_x).transpose()
    _y = np.mat(_y)
    print(_x.shape, _y.shape)
    return _x, _y


def init_theta():
    global theta
    # theta = np.zeros(10, 784)
    for i in range(10):
        d = []
        for j in range(784):
            d.append(0.0)
        theta.append(d)
    theta = np.mat(theta)  # shape: 10 * 784
    print(theta.shape)


def probability(i, j):
    a = np.exp(theta[j:j+1, 0:] * x[0:, i:i+1])
    b = 0
    global k
    for n in range(k):
        b += np.exp(theta[n:n+1, 0:] * x[0:, i:i+1])
    return float(a/b)


def indicator(i, j):
    if y[i:i+1, 0:] == j:
        return 1
    return 0


def each(j):
    _sum = np.zeros([784, 1])
    for i in range(m):
        _sum += x[0:, i:i+1] * (indicator(i, j) - probability(i, j))
    # return (alpha * _sum / m).transpose()
    return alpha * (np.mat(_sum / m).transpose() - weight_lambda * theta[j:j+1, 0:])


def train():
    global theta
    for i in range(times):
        tmp = []
        for j in range(k):
            tmp.append(each(j).tolist()[0])
        theta += np.mat(tmp)
    return theta


def test():
    s = ''
    global theta
    count = 0
    for i in range(tm):
        ip = []
        _x = px[0:, i:i+1]
        for n in range(k):
            a = np.exp(theta[n:n+1, 0:] * _x)
            b = 0
            for _m in range(k):
                b += np.exp(theta[_m:_m+1, 0:] * _x)
            p = a / b
            ip.append(p)
        v = py[i:i+1, 0:]
        _max = ip[0:1]
        index = 0
        for n in range(k):
            if ip[n:n+1] > _max:
                _max = ip[n:n+1]
                index = n
        w = '%dth: predict is %d, v id %d. %s' % (i, index, v, index == v)
        print(w)
        s += w + '\n'
        if index == v:
            count += 1
    print('rate:%f' % (count / tm))  # 0.6804       5:0.687600     7:0.690500    10:0.694700    15:0.702700
    with codecs.open('./test.txt', mode='w') as f:
        f.write(s)


if __name__ == '__main__':
    init_theta()
    x, y = read('F:/Programs/Machine Learning/mnist/pics')  # x.shape: 784 X 60000   y.shape: 60000 X 1
    px, py = read('F:/Programs/Machine Learning/mnist/test')  # px.shape: 784 X 10000   y.shape: 10000 X 1
    start = datetime.datetime.now()
    theta = train()
    end = datetime.datetime.now()
    # print(theta)
    np.mat(theta).dump('./theta.txt')
    test()
    print('training time:', end - start)