import math
import random as r
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as cplt

# data = [(np.array(list(map(float, i[38:].split()))) / 256., np.array(list(map(float, i[8:28].split())))) for i in open('mnist.txt').read().strip().split('\n')]
# pickle.dump(data, open('mnist.pickle', 'wb'))

data = pickle.load(open('mnist.pickle', 'rb'))

training_data = data[:50000]
test_data = data[50000:]

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
    def __init__(self, struct):
        # structure is a tuple with entries representing no. of nodes in each layer
        self.struct = struct
        self.n_layers = len(struct) - 2
        self.input = np.zeros(struct[0])
        self.out = np.zeros(struct[-1])
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
            print("Result\n_____________\n")
            for i in range(len(L)):
                print(str(i) + ":", L[i])
            print("_____________\n")

        return L

    def backprop(self, x, y):
        # Init deltas to 0
        dweight = [np.zeros((w, v)) for v, w in zip(self.struct[:-1], self.struct[1:])]
        dbias = [np.zeros(v) for v in self.struct[1:-1]]
        dactivation = [np.zeros(v) for v in self.struct[1:-1]] # <-- [1:]

        L = [x]
        for i in range(self.n_layers):
            L += [sigmoid(np.dot(self.weight[i], L[i]) + self.bias[i])]
        L += [sigmoid(np.dot(self.weight[self.n_layers], L[self.n_layers]))]

        # dactivation[-1] = y - L[-1]

        for layer in L[1::-1]:
            for i, y in enumerate(layer):
                pass

    def loss(self, x, y):
        loss = 0
        for fxi, yi in zip(self.run(x), y):
            loss += math.pow(fxi - yi, 2)

        return loss / len(y)

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

nn = neural_net((28 * 28, 16, 16, 10))
nn.run(training_data[0][0], show=True)
print("Loss:", nn.loss(*training_data[0]))
print("Accuracy:", str(nn.test(test_data) * 100) + "%")

nn.backprop(*training_data[0])

# nn.show()
# show_img(training_data[0][0])
