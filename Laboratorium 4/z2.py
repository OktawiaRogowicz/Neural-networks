import math
import random
import numpy as np


def relu(weights):
    weights[weights < 0.0] = 0
    return weights


def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + math.e ** -x)  # mathematically equivalent, but simpler


def sigmoid_deriv(a):
    return a * (1 - a)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def softmax(z):
    e = np.exp(z - np.max(z))
    s = np.sum(e, axis=1, keepdims=True)
    return e/s


class NeuralNetwork:
    def __init__(self, input_number, alpha, batch):
        self.weights = []
        self.input_number = input_number
        self.output_number = input_number
        self.alpha = alpha
        self.batch = batch

        self.counter = 0
        self.matching = 0

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        layer = np.random.uniform(weight_min_value, weight_max_value, size=(n, self.output_number))
        self.weights.append(layer)
        self.output_number = n

    def generate_dropout_vector(self, index):
        return np.random.randint(2, size=(len(self.weights[index])))

    def read_images(self, file_name_images):
        file = open(file_name_images, "rb")
        ba = bytearray(file.read())

        mn = []
        q = []
        c = []
        r = []
        for i in range(4):
            mn.append(ba[i])
            q.append(ba[i + 4])
            r.append(ba[i + 8])
            c.append(ba[i + 12])

        magic_number = int.from_bytes(mn, 'big')
        quantity = int.from_bytes(q, 'big')
        rows = int.from_bytes(r, 'big')
        columns = int.from_bytes(c, 'big')
        return [ba, quantity, rows, columns]

    def read_labels(self, file_name_labels):
        file = open(file_name_labels, "rb")
        ba = bytearray(file.read())

        mn = []
        q = []
        for i in range(4):
            mn.append(ba[i])
            q.append(ba[i + 4])

        magic_number = int.from_bytes(mn, 'big')
        quantity = int.from_bytes(q, 'big')
        return ba

    def file_to_vectors(self, n, q, file_name_images, file_name_labels):
        self.matching = 0
        self.counter = 0

        [baI, quantity, rows, columns] = self.read_images(file_name_images)
        baL = self.read_labels(file_name_labels)

        dropout_vectors = []
        vectors = []
        expected_outputs = []

        for x in range(n):
            for i in range(q):
                dropout_vectors.append(self.generate_dropout_vector(0))

                vectors.append(self.one_picture_to_vector(baI, columns, rows, i))
                label = self.one_label_to_int(baL, i)
                expected_outputs.append(self.calc_exp_output(label))
                if i % 99 == 0 and i != 0:
                    # one batch
                    error = self.predict(np.asarray(vectors), np.asarray(expected_outputs), dropout_vectors)
                    dropout_vectors.clear()
                    vectors.clear()
                    expected_outputs.clear()

    def one_picture_to_vector(self, ba, columns, rows, n):
        vector = np.array(np.zeros(columns * rows))
        for i in range(columns * rows):
            x = columns * rows * n + 16
            vector[i] = ba[x + i]
            vector[i] = ((vector[i] * 100.0) / 255.0) / 100.0

        return vector

    def one_label_to_int(self, ba, n):
        return ba[n + 8]

    def calc_exp_output(self, expected_output):
        expected = np.array(np.zeros(10))
        expected[expected_output] = 1.0

        return expected

    def predict(self, input, expected_output, dropout_vector):
        error = 0.0
        layer_1_weights = self.weights[0]
        layer_2_weights = self.weights[1]

        layer_1_values = np.dot(input, layer_1_weights.transpose())
        layer_1_values = relu(layer_1_values)
        layer_1_values = layer_1_values * dropout_vector
        layer_1_values = layer_1_values * 2
        layer_2_values = np.dot(layer_1_values, layer_2_weights.transpose())

        index = np.argmax(layer_2_values, axis=1)
        index2 = np.argmax(expected_output, axis=1)

        layer_2_delta = layer_2_values - expected_output
        l1v = layer_1_values.copy()
        layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * relu_deriv(l1v)
        layer_1_delta = layer_1_delta * dropout_vector

        layer_2_weight_delta = np.dot(layer_2_delta.transpose(), layer_1_values)
        layer_1_weight_delta = np.dot(layer_1_delta.transpose(), input)

        self.weights[1] = self.weights[1] - np.dot(self.alpha, layer_2_weight_delta)
        self.weights[0] = self.weights[0] - np.dot(self.alpha, layer_1_weight_delta)

        error = error + layer_2_delta ** 2

        for i in range(99):
            if index[i] == index2[i]:
                self.matching += 1
            self.counter += 1

        return error


if __name__ == '__main__':
    nn = NeuralNetwork(784, 0.001, 100)

    # Wersja 1:
    nn.add_layer(40, -0.1, 0.1, "reLu")
    nn.add_layer(10, -0.1, 0.1, "reLu")
    nn.file_to_vectors(350, 1000, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
    nn.file_to_vectors(1, 10000, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)

    # Wersja 2:
    nn.add_layer(100, -0.1, 0.1, "reLu")
    nn.add_layer(10, -0.1, 0.1, "reLu")
    nn.file_to_vectors(350, 10000, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
    nn.file_to_vectors(1, 10000, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)

    # Wersja 3:
    nn.alpha = 0.0001
    nn.add_layer(100, -0.1, 0.1, "reLu")
    nn.add_layer(10, -0.1, 0.1, "reLu")
    nn.file_to_vectors(350, 60000, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
    nn.file_to_vectors(1, 10000, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
