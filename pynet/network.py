# -*- coding:utf-8 -*-
import numpy as np
import logging
import sys
import copy
 
class NetWork:

    def __init__(self, name):
        self.name = name
        self.neurals = []
        self.last_in_dim = None
        self.__init_log()
        self.init_opt = False

    def __init_log(self):
        self.logger = logging.getLogger('log[' + self.name + ']')
        self.log_formatter = logging.Formatter('%(asctime)s %(levelname)-5s: %(message)s')
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)
        

    def add(self, neural):
        if (not neural.is_activation()) and (neural.input_dim() is None):
            neural.set_input_dim(self.last_in_dim)
        if not neural.is_activation():  # dense
            self.last_in_dim = neural.output_dim()
        self.neurals.append(neural)

    def loss(self, f):
        self.loss_function = f

    def optimizer(self, opt):
        self.optimizer_function = opt

    def add_logger_handler(self, log_handler):
        log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(log_handler)

    def __output(self, s):
        self.logger.info('['+self.name+'] '+s)

    def __init_optimizer(self):
        for l in self.neurals:
            if not l.is_activation():
                l.set_optimizer(copy.deepcopy(self.optimizer_function), copy.deepcopy(self.optimizer_function))
        
    # iteration 全数据量的训练次数
    # batch_size 每个 epoch 数据量
    # 所有 epoch 训练一次 算 一次 iteration
    def train(self, x, y, batch_size, iteration=1000):
        if not self.init_opt:
            self.init_opt = True
            self.__init_optimizer()
        self.batch_size = batch_size
        epochs = int(len(y) / batch_size)
        if len(y) % batch_size != 0:
            epochs += 1
        for _ in range(iteration):
            for e in range(epochs):
                start = e * batch_size
                end = (e+1) * batch_size
                train_x, train_y = x[start:end], y[start:end]
                self.__batch(train_x, train_y)
            self.__output('loss: ' + str(self.__loss()))
        
    
    def __batch(self, x, y):
        self.error_weight, self.error_bias = [], []

        # init
        for l in self.neurals:
            a, b = 0, 0
            if not l.is_activation():
                a, b = l.input_dim(), l.output_dim()
            self.error_weight.append(np.zeros([a, b]))
            self.error_bias.append(np.zeros([1,b]))

        # 计算样本
        for e, l in zip(x, y):
            p = self.__forward(e)
            self.__calculate_loss(l, p)
            self.__backprocess()
        
        # 计算平均误差
        for i in range(len(self.neurals)):
            if not self.neurals[i].is_activation():
                self.error_weight[i]/self.batch_size
                self.error_bias[i]/self.batch_size

        # 更新
        for i, l in enumerate(self.neurals):
            if not l.is_activation():
                ew , eb = l.weight_optimizer().grad(self.error_weight[i]), l.bias_optimizer().grad(self.error_bias[i])
                ew = np.clip(ew, -1.0, 1.0)
                eb = np.clip(eb, -1.0, 1.0)
                l.update(ew, eb)

        

    # 每计算一个样本，执行一次
    def __forward(self, x):
        p = x
        for _, l in enumerate(self.neurals):
            l.calculate(p)
            p = l.output()
        return p

    def __calculate_loss(self, y, py):
        self.loss_function.calculate(y, py)

    def __loss(self):
        return self.loss_function.loss()

    def __error(self):
        return self.loss_function.error()

    # 每计算一个样本，执行一次
    def __backprocess(self):
        der_error = self.__error()
        for i in range(len(self.neurals)):
            j = len(self.neurals)-i-1
            if self.neurals[j].is_activation():
                der_error = self.neurals[j].derivate(der_error)
            else:
                a, b = self.neurals[j].derivate(der_error)
                self.error_weight[j] += np.reshape(a, self.error_weight[j].shape)
                self.error_bias[j] += np.reshape(b, self.error_bias[j].shape)
                if j > 0:
                    der_error = np.dot(der_error, self.neurals[j].weight().T)
                

    # x array
    def predict(self, x):
        return np.array([self.__forward(a) for a in x])
