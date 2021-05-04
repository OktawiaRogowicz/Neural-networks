import os
import random
import numpy as np


def relu(weights):
    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0
    return weights


def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class NeuralNetwork:
    def __init__(self):
        self.alpha = 0.01
        self.weights = []
        self.counter = 0
        self.matching = 0

        self.input = []
        self.expected_output = []

        self.output_number = 3
        self.input_number = 3

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        layer = np.zeros((n, self.output_number))
        for i in range(n):
            for j in range(self.output_number):
                layer[i][j] = random.uniform(weight_min_value, weight_max_value)
        self.weights.append(layer)
        self.output_number = n

    def read(self, filename):
        self.counter = 0
        self.matching = 0
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, filename)
        file = open(my_file)
        self.count_series(file)
        file.close()

    def predict(self, input, expected_output):
        error = 0.0
        layer_1_weights = self.weights[0]
        layer_2_weights = self.weights[1]

        layer_1_values = np.dot(input, layer_1_weights.transpose())
        layer_1_values = relu(layer_1_values)
        layer_2_values = np.dot(layer_1_values, layer_2_weights.transpose())

        index = np.argmax(layer_2_values)
        index2 = np.argmax(expected_output)


        layer_2_delta = layer_2_values - expected_output
        l1v = layer_1_values.copy()
        layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * relu_deriv(l1v)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.weights[1] = self.weights[1] - np.dot(self.alpha, layer_2_weight_delta)
        self.weights[0] = self.weights[0] - np.dot(self.alpha, layer_1_weight_delta)

        error = error + layer_2_delta ** 2

        if index == index2:
            self.matching += 1
        self.counter += 1

        return error

    @staticmethod
    def calc_exp_output(expected_output):
        if expected_output == 1:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif expected_output == 2:
            return np.array([0.0, 1.0, 0.0, 0.0])
        elif expected_output == 3:
            return np.array([0.0, 0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 0.0, 1.0])

    def count_series(self, file):
        for line in file:
            fields = line.strip().split()

            input = np.array([float(fields[0]), float(fields[1]), float(fields[2])])
            expected_output = self.calc_exp_output(float(fields[3]))
            self.input.append(input)
            self.expected_output.append(expected_output)

    def calculate(self, n):
        error = 0.0
        for j in range(n):
            for i in range(len(self.input)):
                error = error + self.predict(self.input[i], self.expected_output[i])
            error = 0.0


if __name__ == '__main__':
    test_colors = 'test_colors.txt'
    training_colors = 'training_colors.txt'
    nn = NeuralNetwork()
    nn.read(training_colors)
    nn.add_layer(100, -0.1, 0.1, "reLu")
    nn.add_layer(4, -0.1, 0.1, "reLu")
    nn.calculate(50)
    print("Counter: ", nn.counter, "Matching: ", nn.matching)
    print("-----------------------TEST-------------------------")
    nn.read(test_colors)
    nn.calculate(1)
    print("Counter: ", nn.counter, "Matching: ", nn.matching)