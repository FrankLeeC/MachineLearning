import numpy as np
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()

 # calculate distance between a and b
def dist(a, b):
    return np.linalg.norm((a-b), ord=2)

def read():
    rs = []
    with open('./data1.txt', 'r', encoding='utf-8') as file:
        line = ''
        while True:
            line = file.readline()
            if not line:
                break
            ds = line.split(',')
            p = Data(float(ds[0]),float(ds[1]))
            rs.append(p) 
    return rs

class Data:

    def __init__(self, a, b):
        self.list = []
        self.list.append(a)
        self.list.append(b)

    def get_data(self):
        return self.list


class Cluster:

    def __init__(self, data, center):
        self.data = data
        self.center = center

    def get_data(self):
        return self.data

    def get_center(self):
        return self.center

    def get_variance(self):
        return np.sum([x for x in[dist(y.get_data(), self.center) for y in self.data]]) / self.data.__len__()


class KMean:

    def __init__(self, data):
        self.data = data
        self.label = {}  # data -> label
        self.result = {}  # label -> data
        self.c = {}  # cluster center
        self.init_label()
        self.init_center()
    
    def init_label(self):
        for d in self.data:
            self.label[d] = -1

    def init_center(self):
        a = [x.get_data() for x in self.data]
        
        amin = min(np.asmatrix(a)[:, 0:1])
        amax = max(np.asmatrix(a)[:, 0:1])
        bmin = min(np.asmatrix(a)[:, 1:2])
        bmax = max(np.asmatrix(a)[:, 1:2])
        self.c[0] = np.concatenate((np.random.randint(amin, amax, size=1), 
            np.random.randint(bmin, bmax, size=1)))
        self.c[1] = np.concatenate((np.random.randint(amin, amax, size=1), 
            np.random.randint(bmin, bmax, size=1)))

    def cluster(self, data):
        change = False
        current_cluster = {}    
        for d in data:
            label = self.get_label(d)
            if label != self.label[d]:
                self.label[d] = label
                change = True
            arr = current_cluster.get(label, [])
            arr.append(d.get_data())
            current_cluster[label]  = arr
        return change, current_cluster

    def calculate_center(self, arr):
        return np.sum(arr, axis=0) / len(arr)

    def regenerate_center(self, data):
        for k in data:
            p = data[k]
            center = self.calculate_center(p)
            self.c[k] = center

    def call(self):
        iter = 0
        f = True
        clt = None
        while iter < 5000:
            f, clt = self.cluster(self.data)
            self.regenerate_center(clt)
            iter += 1
            if not f:
                break
        rs = {}
        for k in clt:
            center = self.c[k]
            data = [Data(x[0], x[1]) for x in clt[k]]
            cluster = Cluster(data, center)
            rs[k] = cluster
        return rs

    def get_label(self, data):
        rs = []
        for k in self.c:
            rs.append(dist(data.get_data(), self.c[k]))
        if rs.__len__() < 1:
            return 0
        return np.argmin(rs)


def get_split_data(m):
    v = 0
    k = None
    for label in m:
        clt = m[label]
        tmp = clt.get_variance()
        if tmp > v:
            v = tmp
            k = label
    return k, m[k]

def main():
    k = 4
    data = read()
    current_clt = 1
    rs = {0: Cluster(data, np.mean([x.get_data() for x in data]))}  # label -> cluster
    while current_clt < k:
        label, cd = get_split_data(rs)
        clt = KMean(cd.get_data())
        newClt = clt.call()
        rs.pop(label)
        l = 0
        if rs.__len__() > 0:
            l = max(rs.keys())+1
        for j in newClt:
            rs[l+j] = newClt[j]
        current_clt += 1        
    plot(rs)

def plot(data):
    fig, ax = plt.subplots()
    counter = 0
    for k in data:
        color = 'blue'
        if counter == 0:
            color = 'red'
        elif counter == 1:
            color = 'orange'
        elif counter == 2:
            color = 'yellow'
        for d in data[k].get_data():
            ax.scatter(d.get_data()[0], d.get_data()[1], c=color, edgecolors='none')
        counter += 1
    for k in data:
        ax.scatter(data[k].get_center()[0], data[k].get_center()[1], color='black')
    plt.show()

if __name__ == '__main__':
    main()