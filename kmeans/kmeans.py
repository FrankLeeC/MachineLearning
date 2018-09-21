import numpy as np
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()



c1 = np.random.randint(20, 100, size=[1, 2])
c2 = np.random.randint(20, 100, size=[1, 2])
c3 = np.random.randint(20, 100, size=[1, 2])
c = {}
c[0] = c1
c[1] = c2
c[2] = c3

class Pair:

    def __init__(self, a, b):
        self.list = []
        self.list.append(a)
        self.list.append(b)

    def get_data(self):
        return self.list


result = {}

def read():
    rs = []
    with open('./data1.txt', 'r', encoding='utf-8') as file:
        line = ''
        while True:
            line = file.readline()
            if not line:
                break
            ds = line.split(',')
            p = Pair(float(ds[0]),float(ds[1]))
            rs.append(p) 
            result[p] = -1
    return rs


# calculate distance between a and b
def dist(a, b):
    return np.linalg.norm((a-b), ord=2)


def get_label(data):
    rs = []
    for label in c:
        rs.append(dist(data.get_data(), c[label]))
    return np.argmin(rs)


def cluster(data):
    change = False
    current_cluster = {}    
    for d in data:
        label = get_label(d)
        if label != result[d]:
            result[d] = label
            change = True
        arr = current_cluster.get(label, [])
        arr.append(d.get_data())
        current_cluster[label]  = arr
    return change, current_cluster


def calculate_center(arr):
    return np.sum(arr, axis=0) / len(arr)


def regenerate_center(data):
    for k in data:
        p = data[k]
        center = calculate_center(p)
        c[k] = center


def plot():
    fig, ax = plt.subplots()
    for k in result:
        color = 'blue'
        if result[k] == 0:
            color = 'red'
        elif result[k] == 1:
            color = 'orange'
        ax.scatter(k.get_data()[0], k.get_data()[1], c=color, edgecolors='none')
    for k in c:
        ax.scatter(c[k][0], c[k][1], color='black')
    ax.grid(True) 

    plt.show()


def main():
    data = read()
    iter = 0
    f = True
    while iter < 5000:
        f, clt = cluster(data)
        regenerate_center(clt)
        iter += 1
        if not f:
            break
    plot()

if __name__ == '__main__':
    main()