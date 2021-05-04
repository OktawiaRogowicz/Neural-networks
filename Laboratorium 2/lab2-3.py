import numpy as np


class NeuralNetwork:
    def __init__(self, input, weights, expected_output, alpha):
        self.layers = []
        self.input = input
        self.weights = weights
        self.expected_output = expected_output
        self.alpha = alpha

    def one_series(self, n):
        A = self.input[n]
        B = self.weights[0]
        C = A.dot(B.transpose())
        prediction = C
        # print("C: \n", C, "\n\n")

        error = pow(prediction - self.expected_output[n], 2)
        delta = prediction - self.expected_output[n]
        # print("delta: \n", delta)
        weight_delta = np.outer(delta, self.input[n])
        self.weights[0] = self.weights[0] - weight_delta * self.alpha
        # print("wd: \n", weight_delta, "\nweights: \n", self.weights)
        # print("error: \n", error, "\n")
        print(self.weights[0])
        return error

    def count_series(self, n):
        for j in range(n):
            error = 0.0
            for i in range(len(self.input)):
                error = error + self.one_series(i)
            print(error, "\n")
            print(np.sum(error))



if __name__ == '__main__':
    # ZADANIE 3
    weights = np.array([[[0.1, 0.1, -0.3],
                         [0.1, 0.2, 0.0],
                         [0.0, 1.3, 0.1]],
                        [[0.1, 0.1, -0.3],
                         [0.1, 0.2, 0.0],
                         [0.0, 1.3, 0.1]]])
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])
    expected_output = [[0.1, 1, 0.1],
                       [0, 1, 0],
                       [0, 0, 0.1],
                       [0.1, 1, 0.2]]
    alpha = 0.01
    nn = NeuralNetwork(input, weights, expected_output, alpha)
    nn.count_series(50)
