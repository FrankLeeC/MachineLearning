# -*- coding:utf-8 -*-
import numpy as np
import logging
import sys
 
class NetWork:

    def __init__(self, name):
        self.name = name
        self.layers = []
        self.dense_layers = []  # 哪几层是Dense层
        self.layer_count = 0  # 总层数

    def __init_log(self):
        self.logger = logging.getLogger('Neural NetWork')
        self.log_formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)
        

    def add(self, layer):
        if not layer.__is_activation():
            self.last_in_dim = layer.output_dim()
            if not layer.input_dim():
                layer.set_input_dim(self.last_in_dim)
        else:
            self.dense_layers.append(self.layer_count)
        layer.set_index(self.layer_count)
        self.layer_count += 1
        self.layers.append(layer)

    def loss(self, f):
        self.loss_function = f

    def optimizer(self, opt):
        self.optimizer_function = opt

    def add_logger_handler(self, log_handler):
        log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(log_handler)

    def __output(self, s):
        self.logger.info(s)
        
    # iteration 全数据量的训练次数
    # batch_size 每个 epoch 数据量
    # 所有 epoch 训练一次 算 一次 iteration
    def train(self, x, y, batch_size, iteration=1000):
        self.batch_size = batch_size
        epochs = len(y) / batch_size
        if len(y) % batch_size != 0:
            epochs += 1
        for _ in range(iteration):
            for e in range(epochs):
                start = e * batch_size
                end = (e+1) * batch_size
                train_x, train_y = x[start:end], y[start:end]
                self.__batch(train_x, train_y)
        
    
    def __batch(self, x, y):
        self.error_weight, self.error_bias = [], []

        # init
        for l in range(self.layers):
            a, b = l.__input_dim(), l.__output_dim()
            self.error_weight.append(np.zeros([a, b]))
            self.error_bias.append(np.zeros(b))

        # 计算样本
        for e, l in range(zip(x, y)):
            p = self.__forward(e)
            self.__calculate_loss(l, p)
            self.__backprocess()

        self.__output('loss: ' + self.__loss())
        
        # 计算平均误差
        for i in range(len(self.layers)):
            self.error_weight[i]/self.batch_size
            self.error_bias[i]/self.batch_size

        # 更新
        for i, l in enumerate(self.layers):
            if not l.__is_activation():
                ew , eb = self.optimizer_function(self.error_weight[i]), self.optimizer_function(self.error_bias[i])
                l.__update(ew, eb)

        

    # 每计算一个样本，执行一次
    def __forward(self, x):
        p = x
        for _, l in enumerate(self.layers):
            p = l.__calculate(p)
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
        for i in range(len(self.layers)):
            j = len(self.layers)-i-1
            if self.layers[j].__is_activation():
                der_error = self.layers[j].__derivate(der_error)
            else:
                a, b = self.layers[j].__derivate(der_error)
                self.error_weight[j] += a
                self.error_bias[j] += b
                if j > 0:
                    der_error = np.dot(der_error, self.layers[j].__weight().T)
                

    # x array
    def predict(self, x):
        return [self.__forward(a) for a in x]