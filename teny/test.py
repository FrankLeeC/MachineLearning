# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import os
import loss
import network
import neural
import optimizer

def one_hot(n, target):
    return np.array(np.eye(n)[target])

def __convert(fname):
    m = np.where(np.array(Image.open(fname))>=50, 255, 0)/2000
    x, y = m.shape
    m = np.reshape(m, (1, x * y))
    return m

def load(path, count):
    labels = range(10)
    x, y = [], []
    for _, l in enumerate(labels):
        files = os.listdir(path+str(l)+"/")
        for i, f in enumerate(files):
            if (count > 0 and i < count) or count < 0:
                x.append(__convert(path+str(l)+"/"+f))
                y.append(one_hot(10, l))
    return x, y

def load_training_set():
    return load('/Users/frank/Pictures/mnist/train/data_gray/', 10)
    # return [__convert('/Users/frank/Pictures/mnist/train/data_gray/5/10030.png')], np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

def load_test_set():
    return load('/Users/frank/Pictures/mnist/test/data_gray/', 10)

def get_network():
    n = network.NetWork('network')
    n.add(neural.Dense(10, input_dim=784))
    n.add(neural.Softmax())
    n.loss(loss.CategoryCrossEntropy())
    n.optimizer(optimizer.Adagrad())
    return n

def run():
    x, y = load_training_set()
    n = get_network()
    n.train(x, y, 30, iteration=10000)
    x, y = load_test_set()
    y1 = n.predict(x)
    same = 0
    for i, l in enumerate(y):
        if np.argmax(l) == np.argmax(y1[i]):
            same += 1
        print(np.argmax(l), np.argmax(y1[i]))
    print(same/len(y))

    
if __name__ == "__main__":
    run()