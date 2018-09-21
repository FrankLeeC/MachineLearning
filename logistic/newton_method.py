# -*-encode=utf-8-*-

import codecs
import numpy as np


x, y = [], []
px, py = [], []


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_matrix(x, theta):
    d = []
    for i, e in enumerate(x):
        d.append([sigmoid(float(np.mat(e) * np.mat(theta)))])
    return np.mat(d)


def read(path='./data1.txt'):
    global x, y, px, py
    with codecs.open(path, mode='r') as file:
        while True:
            line = file.readline()
            if not line:
                x, y = np.mat(x)[0: 100], np.mat(y)[0: 100]
                px, py = np.mat(x)[100:], np.mat(y)[100:]
                break
            ds = line.split(',')
            d = [float(ds[0]), float(ds[1]), 1]
            x.append(d)
            y.append([float(ds[2])])


def train():
    global x, y
    theta = np.zeros((3, 1))
    for i in range(200):
        loss = y - sigmoid_matrix(x, theta)
        direv = np.matrix(x).transpose() * loss
        h = np.matrix(x.transpose() * x * (sigmoid_matrix(x, theta).transpose() * (1 - sigmoid_matrix(x, theta)))[0, 0]).I
        theta = theta - h * direv
    print(theta)
    return theta


read()
train()
