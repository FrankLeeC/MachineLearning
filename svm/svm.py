# -*- coding:utf-8 -*-

import numpy as np
import random

class SVM:

    def __init__(self, kernel='rbf', C=10.0, gamma=0.5, p=2):
        '''
        kernel: rbf 高斯核函数  linear 线性核函数  polynomial  多项式
        C: penalty term 惩罚因子
        gammar: hyperparameter of rbf/polynomial kernel  用于 rbf/polynomial 的超参数
        p: hyperparameter of polynomial kernel  用于 polynomial 的超参数
        '''
        if C <= 0.0:
            C = 1.0
        self.b = 0
        self.alpha = list()
        self.kernel_type = kernel
        self.C = C
        self.epsilon = 1e-3  # SVM 目标函数下降的范围
        self.tol = 1e-3  # KKT 误差
        self.gamma = gamma
        self.p = p
        self.error_list = list()

    def fit(self, x, y):
        '''
        x: count by dimension matrix
        y: 1 by count vector/list
        '''
        self.__prepare(x, y)
        self.__train()

    def predict(self, x):
        '''
        sign of decision function
        '''
        return np.sign(self.__decision(x, self.alpha))

    def __prepare(self, x, y):
        '''
        x: count by dimension matrix
        y: 1 by count vector/list
        '''
        self.train_x = np.asmatrix(x)
        self.train_y = np.asarray(y)
        self.count, self.dimension = np.shape(self.train_x)
        self.w = np.zeros((1, self.dimension))
        self.alpha = np.zeros(self.count)
        self.error_list = np.zeros(self.count)

    def __calculate_w(self):
        self.w = np.sum(self.alpha[i] * self.train_y[i] * self.train_x[i] for i in range(len(self.train_y)))

    def __train(self):
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.count):
                    if self.__examine_example(i):
                        num_changed += 1
            else:
                c = np.arange(self.count)
                np.random.shuffle(c)
                for i in c:
                    if self.alpha[i] != 0 and self.alpha[i] != self.C and self.__examine_example(i):
                        num_changed += 1
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

    def __take_step(self, i1, i2):
        if i1 == i2:
            return False
        a1 = self.alpha[i1]
        a2 = self.alpha[i2]
        x1 = self.train_x[i1]
        x2 = self.train_x[i2]
        y1 = self.train_y[i1]
        y2 = self.train_y[i2]
        e1 = self.__get_error(i1)
        e2 = self.__get_error(i2)
        s = y1 * y2
        if y1 != y2:
            l = max(0, a2 - a1)
            h = min(self.C, self.C + a2 - a1)
        else:
            l = max(0, a2 + a1 - self.C)
            h = min(self.C, a1 + a2)
        if l == h:
            return False
        
        k11 = self.__kernel(x1, x1)
        k22 = self.__kernel(x2, x2)
        k12 = self.__kernel(x1, x2)
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            new_a2 = a2 + y2 * (e1 - e2) / eta
            if new_a2 < l:
                new_a2 = l
            elif new_a2 > h:
                new_a2 = h
        else:
            f1 = y1 * (e1 + self.b) - a1 * k11 - s * a2 * k12
            f2 = y2 * (e2 + self.b) - s * a1 * k12 - a2 * k22
            l1 = a1 + s * (a2 - l)
            h1 = a1 + s * (a2 - h)
            l_obj_value = l1*f1 + l*f2 + 0.5 * (l1**2)*k11 + 0.5*(l**2)*k22 + s*l*l1*k12
            h_obj_value = h1*f1 + h*f2 + 0.5 * (h1**2)*k11 + 0.5*(h**2)*k22 + s*h*h1*k12
            if l_obj_value < h_obj_value - self.epsilon:
                new_a2 = l
            elif h_obj_value < l_obj_value - self.epsilon:
                new_a2 = h
            else:
                new_a2 = a2

        if new_a2 < 1e-8:
            new_a2 = 0
        elif new_a2 > self.C - 1e-8:
            new_a2 = self.C
        
        if abs(new_a2 - a2) < self.epsilon * (new_a2 + a2 + self.epsilon):
            return False
        new_a1 = a1 + s * (a2 - new_a2)

        b1 = e1 + y1 * (new_a1-a1) * k11 + y2 * (new_a2-a2) * k12 + self.b
        b2 = e2 + y1 * (new_a1-a1) * k12 + y2 * (new_a2-a2) * k22 + self.b
        if 0 < new_a1 and new_a1 < self.C:
            new_b = b1
        elif 0 < new_a2 and new_a2 < self.C:
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0

        db = new_b - self.b
        self.b = new_b

        if self.kernel_type == 'linear':
            self.w += y1 * float(new_a1 - a1) * x1 + y2 * float(new_a2 - a2) * x2

        d1 = y1*(new_a1 - a1)
        d2 = y2*(new_a2 - a2)
        for i in range(self.count):
            if 0 < self.alpha[i] < self.C:
                xi = self.train_x[i]
                self.error_list[i] += d1 * self.__kernel(x1, xi) + d2 * self.__kernel(x2, xi) - db
        
        self.error_list[i1] = 0
        self.error_list[i2] = 0
        self.alpha[i1] = float(new_a1)
        self.alpha[i2] = float(new_a2)
        return True

    def __examine_example(self, i2):
        y2 = self.train_y[i2]
        a2 = self.alpha[i2]
        e2 = self.__get_error(i2)
        r2 = e2 * y2
        if (r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
            if len(np.where((self.alpha!=0)&(self.alpha!=self.C))) > 1:
                if e2 > 0:
                    i1 = np.argmin(self.error_list)
                elif e2 < 0:
                    i1 = np.argmax(self.error_list)
                else:
                    a = np.argmin(self.error_list)
                    b = np.argmax(self.error_list)
                    if abs(self.error_list[a]) > abs(self.error_list[b]):
                        i1 = a
                    else:
                        i1 = b
                if self.__take_step(i1, i2):
                    return True
            c = np.arange(self.count)
            np.random.shuffle(c)
            for i in c:
                if self.alpha[i] != 0 and self.alpha[i] != self.C and self.__take_step(i, i2):
                    return True
            np.random.shuffle(c)
            for i in c:
                if self.__take_step(i, i2):
                    return True
        return False

    def get_model(self):
        self.__calculate_w()
        return self.w, self.b
    
    def __decision(self, x, alpha):
        '''
        g(x_i) = sum_1^n alpha_i * y_i * K(x_i, x) - b
        '''
        if self.kernel_type == 'linear':
            return np.dot(self.w, x.T) - self.b
        t = 0
        for i in range(self.count):
            t = t + alpha[i] * self.train_y[i] * self.__kernel(x, self.train_x[i])
        return t - self.b
    
    def __get_error(self, i):
        if 0 < self.alpha[i] < self.C:
            return self.error_list[i]
        else:
            return self.__decision(self.train_x[i], self.alpha) - self.train_y[i]

    def __observe_function(self, alpha):
        '''
        SVM 的目标函数
        0.5 * (sum_i^nsum_1^jalpha_i * alpha_j * y_i * y_j * K(x_i, x_j)) - sum_1_n * alpha_i
        '''
        y = np.asmatrix(self.train_y)
        x = np.asmatrix(self.train_x)
        a = np.asmatrix(alpha)
        return 0.5 * np.sum(np.multiply(np.multiply(a.T*a, y.T*y), self.__kernel(x, x))) - np.sum(a)

    def __kernel(self, x1, x2):
        '''
        kernel trick
        rbf
        linear
        polynomial
        '''
        if self.kernel_type == 'rbf':
            return self.__rbf(x1, x2)
        elif self.kernel_type == 'polynomial':
            return self.__polynomial(x1, x2)
        elif self.kernel_type == 'linear':
            return self.__linear(x1, x2)

    def __rbf(self, x1, x2):
        '''
        rbf kernel:
        K(x1, x2) = exp[-gamma||x1-x2||^2]
        '''
        x1 = np.asmatrix(x1)
        x2 = np.asmatrix(x2)
        r, _ = np.shape(x1)
        r2, _ = np.shape(x2)
        if r == r2 == 1:
            return np.exp(-self.gamma*np.linalg.norm(x1-x2)**2)
        else:
            l = list()
            for i in r:
                tx1 = x1[i]
                tmp = tx1 - x2
                l2 = list()
                for j in r:
                    t = tmp[j]
                    l2.append(np.exp(-self.gamma*np.linalg.norm(t)**2))
                l.append(l2)
            return np.asmatrix(l)

    def __polynomial(self, x1, x2):
        '''
        polynomial kernel:
        K(x1, x2) = (gammar * x1 * x2 + 1)^p
        '''
        x1 = np.asmatrix(x1)
        x2 = np.asmatrix(x2)
        r, _ = np.shape(x1)
        r2, _ = np.shape(x2)
        if r == r2 == 1:
            return (self.gamma * np.dot(x1, x2.T) + 1) ** self.p
        else:
            l = list()
            for i in r:
                tx1 = x1[i]
                l2 = list()
                for j in r:
                    tx2 = x2[j]
                    l2.append((self.gamma * np.dot(tx1, tx2.T) + 1) ** self.p)
                l.append(l2)
            return np.asmatrix(l)

    def __linear(self, x1, x2):
        return np.dot(x1, x2.T) + self.b
