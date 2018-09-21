# -*-encode=utf-8-*-
import codecs

import datetime
import numpy as np
from scipy.misc import imread
import os

theta, x, y, px, py = [], [], [], [], []
k = 10
alpha = 0.001
m = 60000
tm = 10000
times = 600
weight_lambda = 0.08


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


def prob(j):
    global y
    molecule = []
    denominator = []
    indicator = []
    for i in range(m):
        molecule.append([np.mat(np.exp(theta[j:j+1, 0:] * x[0:, i:i+1])).tolist()[0][0]])  # shape: 60000 X 1

        d = theta * x[0:, i:i + 1]
        denominator.append([np.mat(np.sum(np.exp(d), axis=1)).tolist()[0][0]])  # shape: 60000 X 1

        if y[i:i+1, 0:] == j:
            indicator.append([1])
        else:
            indicator.append([0])
    return np.mat(indicator) - np.mat(molecule) / np.mat(denominator)  # shape: 60000 X 1


def gradient():
    global x
    gra = []
    for j in range(k):
        t = (np.mat(x) * prob(j)).transpose()
        d = []
        for _t in t.tolist()[0]:
            d.append(_t)
        gra.append(d)
    return np.mat(gra)  # shape: 10 X 784


def train():
    global theta
    for num in range(times):
        theta += alpha * ((gradient() / m) - weight_lambda * np.mat(theta))  # 衰减权重
    np.mat(theta).dump('./theta_matrix.txt')


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
            p = np.mat(a) / np.mat(b)
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
    print('rate:%f' % (count / tm))  # 3:0.713900  10:0.755600   30:0.778900  200:0.830900   300:0.840200   600:0.848800
    with codecs.open('./test_matrix.txt', mode='w') as f:
        f.write(s)


if __name__ == '__main__':
    init_theta()
    x, y = read('F:/Programs/Machine Learning/mnist/pics')  # x.shape: 784 X 60000   y.shape: 60000 X 1
    px, py = read('F:/Programs/Machine Learning/mnist/test')  # px.shape: 784 X 10000   y.shape: 10000 X 1
    start = datetime.datetime.now()
    train()
    end = datetime.datetime.now()
    print('cost', end - start)
    # print(theta)
    test()