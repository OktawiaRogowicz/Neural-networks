import numpy as np
import math


class ReLU2:
    def __init__(self, input, weights, expected_output):
        self.input = input
        self.weights = weights
        self.alpha = 0.01
        self.expected_output = expected_output

    def relu(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if weights[i][j] < 0:
                   weights[i][j] = 0
        return weights

    def relu_deriv(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def count_series(self, n):
        error = 0.0
        for j in range(n):
            error = self.calc()
            print("error: ", sum(error))

    def calc(self):
        values = [None] * len(weights)
        before = self.input
        for i in range(len(self.weights)):
            values[i] = before.dot(self.weights[i].transpose())
            values[i] = self.relu(values[i])
            before = values[i]

        delta = [None] * len(weights)
        delta[len(self.weights) - 1] = values[len(self.weights) - 1] - expected_output
        error = pow(delta[len(self.weights) - 1], 2)
        for i in reversed((range(len(self.weights) - 1))):
            delta[i] = delta[i+1].dot(self.weights[i+1])
            rd = np.copy(values[i])
            delta[i] = delta[i] * self.relu_deriv(rd)

        weightDelta = []
        for i in range(len(self.weights)):
            wd = []
            for j in range(len(delta[i])):
                if i >= 1:
                    wd = np.outer(delta[i][j], values[i-1][j])
                else:
                    wd = np.outer(delta[i][j], self.input[j])
                self.weights[i] = self.weights[i] - (self.alpha * wd)
                print("i: ", i, "j: ", j, "\n", self.weights[i])

        return error

    # def calc(self):
    #     error = 0.0
    #     layer_1_weights = self.weights[0]
    #     layer_2_weights = self.weights[1]
    #
    #     for i in range(len(self.input)):
    #         layer_1_values = np.dot(self.input[i], layer_1_weights.transpose())
    #         layer_1_values = self.relu(layer_1_values)
    #         layer_2_values = np.dot(layer_1_values, layer_2_weights.transpose())
    #
    #         layer_2_delta = layer_2_values - expected_output[i]
    #         l1v = layer_1_values.copy()
    #         layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * self.relu_deriv(l1v)
    #
    #         layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
    #         layer_1_weight_delta = np.outer(layer_1_delta, input[i])
    #
    #         self.weights[1] = self.weights[1] - np.dot(self.alpha, layer_2_weight_delta)
    #         self.weights[0] = self.weights[0] - np.dot(self.alpha, layer_1_weight_delta)
    #
    #         error = error + layer_2_delta ** 2
    #
    #     print("error: ", sum(error))


if __name__ == '__main__':
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])
    weights = np.array([[[0.1, 0.2, -0.1],
                        [-0.1, 0.1, 0.9],
                        [0.1, 0.4, 0.1]],
                        [[0.3, 1.1, -0.3],
                        [0.1, 0.2, 0.0],
                        [0.0, 1.3, 0.1]]])
    expected_output = np.array([[0.1, 1, 0.1],
                                [0, 1, 0],
                                [0, 0, 0.1],
                                [0.1, 1, 0.2]])
    r = ReLU2(input, weights, expected_output)
    r.count_series(50)
