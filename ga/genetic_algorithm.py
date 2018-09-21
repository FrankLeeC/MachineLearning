# -*- coding:utf-8 -*-

import math
import random
import copy

population_size = 20  # 群体数目
crossover_propability = 0.7  # 交叉繁衍概率
mutate_propability = 0.05  # 变异概率

def init_population():
    '''
    初始化群体
    '''
    population = list()  # 群体
    for _ in range(population_size):
        population.append(to_bit(random.randint(0, 100)))
    return population


def to_bit(n):
    '''
    十进制转8位二进制数组
    '''
    b = bin(n)[2:].rjust(8, '0')
    r = list()
    for i in range(len(b)):
        r.append(int(b[i]))
    return r 


target = to_bit(17)
def cosine(a):
    '''
    fitness function 
    '''
    def multiply(x, y):
        m = 0.0
        for i, e in enumerate(x):
            m += e*y[i]
        return m
    m = multiply(a, target)
    norm_a = multiply(a, a)
    norm_b = multiply(target, target)
    if norm_b == 0:
        if norm_a == 0:
            return 1
        return -1
    return m / math.sqrt(norm_a * norm_b)


def fit(population, func):
    return [func(e) for e in population]


def select(population, fitness):
    '''
    Roulette Wheel Selection
    轮盘赌选择
    return:
    parent1 chromosome, parent2 chromosome
    '''
    fit_list = copy.deepcopy(fitness)
    fit_sum = sum(fit_list)
    # r = random.uniform(0.0, fit_sum)
    r = random.randint(0, int(fit_sum))
    cache = list()
    def choose(r):
        tmp = 0.0
        for i, e in enumerate(fit_list):
            tmp += e
            if tmp >= r and i not in cache:
                return i
    idx1 = choose(r) 
    cache.append(idx1)
    p1 = population[idx1]

    idx2 = choose(r)
    if idx2:
        p2 = population[idx2]
    else:
        p2 = population[idx1]
    return p1, p2


def crossover(p1, p2):
    '''
    交叉繁衍是一个概率事件
    倘若发生，选择一个切割点，生成两个新的样本
    否则，返回原本的父样本
    '''
    r = random.random()
    if r > crossover_propability:
        return p1, p2
    r = random.randint(1, len(p1)-1)
    c1 = p1[0:r]
    c1.extend(p2[r:])
    c2 = p2[0:r]
    c2.extend(p1[r:])
    return c1, c2


def mutate(a):
    '''
    变异是一个概率事件
    倘若发生，随机一个 gene 进行变异
    否则不变       
    '''
    r = random.random()
    if r > mutate_propability:
        return a
    p = random.randint(0, 7)
    a[p] = int(not a[p])
    return a


def elitism(population, fitness):
    '''
    精英保留 2 个
    '''
    fitcopy = copy.deepcopy(fitness)
    def find_max():
        maxfit = fitcopy[0]
        maxidx = 0
        for i, e in enumerate(fitcopy):
            if e > maxfit:
                maxfit = e
                maxidx = i
        return maxidx, maxfit
    idx, _ = find_max()
    fitcopy[idx] = float('-inf') 
    idx2, _ = find_max()
    return idx, idx2


def accept(population, fit_list, offspring):
    '''
    将群体中fit值最低的两个改成新的
    '''
    fitness = copy.deepcopy(fit_list)
    cache = list()
    def find_min():
        fit_min = fitness[0]
        min_idx = 0
        for i, e in enumerate(fitness):
            if i not in cache and e < fit_min:
                min_idx = i
                fit_min = e
        return min_idx
    idx1 = find_min()
    cache.append(idx1)
    population[idx1] = offspring[0]
    idx2 = find_min()
    population[idx2] = offspring[1]
    return population


def end(population):
    '''
    只要有一个样本满足目标，就结束
    也可以改为全部满足
    也可以加上没有变化
    '''
    for e in population:
        if cosine(e) != 1.0:
            return False
    return True


def decimal(b):
    '''
    二进制数组转为十进制
    '''
    b = ''.join(str(e) for e in b)
    return int(b, 2)


def main():
    population = init_population()
    print('init: ')
    for e in population:
        print(e, '--->', decimal(e))
    print('--------------')
    count = 1 
    while not end(population):
        count += 1
        fitness = fit(population, cosine)
        p1, p2 = select(population, fitness)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        elitism(population, fitness)
        population = accept(population, fitness, [c1, c2])
    print('generation:%d'%count) 
    print('over: ')
    for e in population:
        print(e, '--->', decimal(e))

if __name__ == '__main__':
    main()
'''
Outline of the Basic Genetic Algorithm
1.[Start] Generate random population of n chromosomes (suitable solutions for the problem)
2.[Fitness] Evaluate the fitness f(x) of each chromosome x in the population
3.[New population] Create a new population by repeating following steps until the new population is complete
    [Selection] Select two parent chromosomes from a population according to their fitness (the better fitness, the bigger chance to be selected)
    [Crossover] With a crossover probability cross over the parents to form a new offspring (children). If no crossover was performed, offspring is an exact copy of parents.
    [Mutation] With a mutation probability mutate new offspring at each locus (position in chromosome).
    [Accepting] Place new offspring in a new population
4.[Replace] Use new generated population for a further run of algorithm
5.[Test] If the end condition is satisfied, stop, and return the best solution in current population
6.[Loop] Go to step 2
'''
