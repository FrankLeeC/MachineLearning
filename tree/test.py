# -*- coding:utf-8 -*-

import decision_tree as dt

def get_data():
    x, y = list(), list()
    f = open('./car.data.txt', encoding='utf-8')
    while True:
        line = f.readline().strip()
        if not line:
            break
        data = line.split(',')
        x.append(data[0:len(data)-1])
        y.append(data[len(data)-1])
    return x, y
        
def get_test():
    x, y = list(), list()
    f = open('./car.test.txt', encoding='utf-8')
    while True:
        line = f.readline().strip()
        if not line:
            break
        data = line.split(',')
        x.append(data[0:len(data)-1])
        y.append(data[len(data)-1])
    return x, y
    

def main():
    x, y = get_data()
    model = dt.DecisionTree([0,1,2,3,4,5])
    model.fit(x, y)
    count = 0
    _sum = 0
    x, y = get_test()
    for i, e in enumerate(x):
        p = model.predict(e)
        # print(p, y[i])
        if p== y[i]:
            count += 1
        _sum += 1
    print(count/_sum)


if __name__ == '__main__':
    main()
