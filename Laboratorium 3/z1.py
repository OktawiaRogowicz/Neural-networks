import numpy as np


class ReLU1:
    def __init__(self, input, weights):
        self.input = input
        self.weights = weights

    def relu(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if weights[i][j] < 0:
                   weights[i][j] = 0
        return weights

    def calc(self):
        first_layer = input

        for i in range(len(self.weights)):
            A = first_layer
            B = self.weights[i]
            C = A.dot(B.transpose())
            if i < len(self.weights) - 1:
                first_layer = self.relu(C)
            else:
                first_layer = C
            print(first_layer)


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

    r = ReLU1(input, weights)
    r.calc()
