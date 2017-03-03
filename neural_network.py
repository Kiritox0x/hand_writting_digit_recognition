import numpy as np
import matplotlib.pyplot as plt
import random
import csv


def get_data():
    train_file = open('data/train.csv', 'rb')
    test_file = open('data/test.csv', 'rb')
    rtr = csv.reader(train_file)
    rte = csv.reader(test_file)
    training_inputs = []
    training_labels = []
    validation_labels = []
    test_data = []
    for i, x in enumerate(rtr):
        if (i == 0):
            continue
        x = map(int, x)
        data = np.array(x[1:]).astype(float) / 255
        label = np.zeros((10, 1))
        label[x[0]][0] = 1
        training_inputs.append(data.reshape(len(data), 1))
        training_labels.append(label)
        validation_labels.append(x[0])
    training_data = zip(training_inputs, training_labels)
    validation_inputs = training_inputs[len(training_inputs) * 4 / 5:]
    validation_data = zip(validation_inputs, validation_labels[len(training_inputs) * 4 / 5:])
    training_data = training_data[:len(training_data) * 4 / 5 - 1]
    for i, x in enumerate(rte):
        if (i == 0):
            continue
        x = map(int, x)
        data = np.array(x).astype(float) / 255
        test_data.append((data.reshape(len(data), 1)))
    return training_data, validation_data, test_data


class Neural_network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        self.W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.num_layers = len(sizes)


    def score(self, a):
        for b, w in zip(self.b, self.W):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self, training_data, step, mini_batch_size, eta, validation_data):
        n_valid = len(validation_data)
        n = len(training_data)
        for j in xrange(step):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print "Step {0}: {1} / {2}".format(j, self.evaluate(validation_data), n_valid)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.W]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.W = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.W, nabla_w)]
        self.b = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.W]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b, self.W):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.W[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.score(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def predict(self, test_data):
        test_results = [np.argmax(self.score(x)) for x in test_data]
        with open('result2.csv', 'w') as csvfile:
            csvfile.write('ImageId,Label\n')
            for i, x in enumerate(test_results):
                csvfile.write(str(i+1) + ',' + str(x) + '\n')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


train_data, valid_data, test_data = get_data()
net = Neural_network([784, 150, 30, 10])
net.gradient_descent(train_data, 30, 10, 1.5, valid_data)
net.predict(test_data)
