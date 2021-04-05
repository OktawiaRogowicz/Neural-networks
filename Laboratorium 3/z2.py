import numpy as np


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
            print(error, "\n")

    def calc(self):
        first_layer = self.input

        for i in range(len(self.weights)):
            A = first_layer
            B = self.weights[i]
            C = A.dot(B.transpose())

            print("A: ", A)
            print("B: ", B)
            print("C: ", C)

            if i < len(self.weights) - 1:
                first_layer = self.relu(C)
            # else:
            #     delta2 = C - self.expected_output
            #     delta1 = delta2.dot(self.weights[i])
            #     delta1 = delta1 * self.relu_deriv(first_layer)
            #     print("delra2: ", delta2)
            #     print(weights[i-1])
            #     weight_delta2 = delta2.dot(weights[i-1].transpose())
            #     print("weight delta2: ", weight_delta2)
            #     weight_delta1 = delta1.dot(input.transpose())
            #     self.weights[i] = self.weights[i] - self.alpha * weight_delta2
            #     self.weights[i-1] = self.weights[i-1] - self.alpha * delta1
            #     error = pow(C - expected_output, 2)
            #     return error


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
