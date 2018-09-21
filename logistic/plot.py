import matplotlib.pyplot as plt


def read():
    data, label = [], []
    file = open("./data1.txt")
    lines = file.readlines()
    for line in lines:
        d = line.strip().split(",")
        data.append([float(d[0]), float(d[1])])
        label.append([int(d[2])])
    return data, label


data, label = read()

positive_arg = []
positive_val = []
negative_arg = []
negative_val = []


for i, e in enumerate(label):
    if e[0] == 0:
        positive_arg.append(float(data[i][0]))
        positive_val.append(float(data[i][1]))
    else:
        negative_arg.append(float(data[i][0]))
        negative_val.append(float(data[i][1]))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.scatter(positive_arg, positive_val, c='blue', label='positive', edgecolors='none')
ax.scatter(negative_arg, negative_val, c='red', label='negative', edgecolors='none')

x_set = []
y_set = []
x = 0.0
for i in range(10000):
    x = x + 0.01
    y = -1.1036 * x + 132.2506
    x_set.append(x)
    y_set.append(y)

ax.scatter(x_set, y_set, c='black', label='predict', edgecolors='none')


ax.legend()
ax.grid(True)

plt.show()