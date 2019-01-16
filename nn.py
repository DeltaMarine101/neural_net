import math
import random as r
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.colors as cplt

# data = [(np.array(list(map(float, i[38:].split()))) / 256., np.array(list(map(float, i[8:28].split())))) for i in open('data/mnist.txt').read().strip().split('\n')]
# pickle.dump(data, open('data/mnist.pickle', 'wb'))

data = pickle.load(open('data/mnist.pickle', 'rb'))

training_data = data[:50000]
test_data = data[50000:]

def time_func(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        print('{:.3f} s'.format(time.time() - t1))
        return res
    return wrapper

def show_img(img, size=28):
    plt.style.use('dark_background')

    cmap = cplt.LinearSegmentedColormap.from_list("", ["k", "w"])

    fig, ax = plt.subplots()
    ax.set_axis_off()
    X = np.reshape(img, (size, size))
    ax.imshow(X, interpolation='nearest', cmap=cmap)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.1, hspace=0.1)

    plt.show()

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class neural_net:
    def __init__(self, struct, lr=.1, rp=0.01):
        # structure is a tuple with entries representing no. of nodes in each layer
        self.struct = struct
        self.n_layers = len(struct) - 2
        self.lr = lr
        self.rp = rp
        # [
        #     [ // layer 1->2
        #         [0, ...], // hidden 1
        #         [0, ...]  // hidden 2
        #     ],
        #     [] // layer 2->3
        # ]

        # Random value initalisation

        self.weight = [(np.random.rand(w, v) * 2 - 1)  / math.sqrt(v) for v, w in zip(struct[:-1], struct[1:])]
        self.bias = [np.random.rand(v) for v in struct[1:-1]]

    def run(self, x, show=False):
        L = x
        for i in range(self.n_layers):
            L = sigmoid(np.dot(self.weight[i], L) + self.bias[i])
        L = sigmoid(np.dot(self.weight[self.n_layers], L))

        if show:
            print("Result\n_________________\n")
            for i in range(len(L)):
                print(str(i) + ":", L[i])
            print("_________________\n")

        return L

    @time_func
    def backprop(self, training):
        # Init deltas to 0
        dweight = [np.zeros((w, v)) for v, w in zip(self.struct[:-1], self.struct[1:])]
        dbias = [np.zeros(v) for v in self.struct[1:-1]]
        dactivation = [np.zeros(v) for v in self.struct[1:]]

        for x, y in training:
            L = [x]
            for i in range(self.n_layers):
                L += [sigmoid(np.dot(self.weight[i], L[i]) + self.bias[i])]
            L += [sigmoid(np.dot(self.weight[-1], L[-1]))]

            dactivation[-1] = (2 / self.struct[-1]) * (L[-1] - y)

            for n in reversed(range(self.n_layers + 1)):
                bias = [np.zeros(self.struct[n])] + self.bias
                for nodej in range(self.struct[n + 1]):
                    deriv = sigmoid(L[n]* self.weight[n][nodej] + bias[n], derivative=True) * dactivation[n][nodej]
                    dweight[n][nodej] += deriv * L[n] # + (1, -1)[L[n][nodei] > 0] * self.rp / len(data)
                    if n > 0:
                        dbias[n - 1] += deriv
                        dactivation[n - 1] += deriv * self.weight[n][nodej]

        self.weight = [x - (y * self.lr) / len(training) for x, y in zip(self.weight, dweight)]
        self.bias = [x - (y * self.lr)  / len(training) for x, y in zip(self.bias, dbias)]

    def loss(self, data):
        return sum([sum(np.square(self.run(i[0]) - i[1])) / len(i[1]) for i in data]) / len(data) #+ self.rp * sum([np.sum(np.square(i)) for i in self.weight]) / (2 * len(data))

    def test(self, test_data):
        n_pass = 0
        for x, y in test_data:
            fx = self.run(x)
            max = fx[0]
            max_i = 0
            for i in range(len(fx)):
                if fx[i] > max:
                    max = fx[i]
                    max_i = i
            if y[max_i] == 1:
                n_pass += 1

        return n_pass / len(test_data)

    def show(self, layer=1):
        plt.style.use('dark_background')

        side = int(math.sqrt(self.struct[layer - 1]))

        cmap = cplt.LinearSegmentedColormap.from_list("", ["#ff704d", "#222222", "#70db70"])

        n_plots = int(math.sqrt(self.struct[layer]))
        fig, axes = plt.subplots(n_plots, n_plots)
        for i in range(n_plots):
            for j in range(n_plots):
                axes[i, j].set_axis_off()
                X = np.reshape(self.weight[layer - 1][i * n_plots + j], (side, side))
                axes[i, j].imshow(X, interpolation='nearest', cmap=cmap)

        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.1, hspace=0.1)
        plt.show()

    def save(self, name='last_model.nn'):
        pickle.dump((self.struct, self.n_layers, self.lr, self.weight, self.bias), open('model/' + name, 'wb'))

    def load(self, name='last_model.nn'):
        self.struct, self.n_layers, self.lr, self.weight, self.bias = pickle.load(open('model/' + name, 'rb'))

nn = neural_net((28 * 28, 16, 16, 10))
# nn.load()
nn.run(training_data[0][0], show=True)
loss = nn.loss(training_data)
print("Initial loss:", loss)
# print("Accuracy:", str(nn.test(test_data) * 100) + "%")

# show_img(training_data[0][0])
# nn.show()

prev = loss
batch = 50
cycles = 100
while True:
    for i, data in enumerate([training_data[n:n + batch] for n in range(0, batch * cycles, batch)]):
        if not i % 10:
            print(i, "Accuracy:", str(nn.test(test_data) * 100) + "%")

        nn.backprop(data)

        loss = nn.loss(training_data[i % 3::3])

        print("(" + str(i + 1) + "/" + str(cycles) + ") Loss:", loss, ['+', '-'][prev > loss])
        prev = loss

        nn.save()

# for i in range(1000):
#     nn.backprop([training_data[0]])
# print("Accuracy:", str(nn.test(test_data) * 100) + "%")
print("Final Accuracy:", str(nn.test(test_data) * 100) + "%")
nn.run(training_data[0][0], show=True)

# nn.show()
# show_img(training_data[0][0])
