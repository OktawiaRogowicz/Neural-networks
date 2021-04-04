import os

# RED [1], GREEN [2], BLUE [3], YELLOW [4]
import numpy as np


def recognise_colour(prediction, expected_output):
    index = np.argmax(prediction)
    if index == 0:
        print("RED")
    elif index == 1:
        print("GREEN")
    elif index == 2:
        print("BLUE")
    else:
        print("YELLOW")
    index2 = np.argmax(expected_output)
    if index == index2:
        print("IT MATCHES!")


class NeuralNetwork:
    def __init__(self, weights):
        self.alpha = 0.01
        self.weights = weights[0]

    def read(self, filename, n):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, filename)
        file = open(my_file)
        self.count_series(n, file)
        file.close()

    def one_series(self, input, expected_output):
        A = input
        # print(A)
        B = self.weights
        # print(B)
        C = A.dot(B.transpose())
        prediction = C
        # print("C: \n", C, "\n\n")
        # print(expected_output)

        recognise_colour(prediction, expected_output)

        error = pow(prediction - expected_output, 2)
        delta = prediction - expected_output
        # print("delta: \n", delta)
        weight_delta = np.outer(delta, input)
        self.weights = self.weights - weight_delta * self.alpha
        # print("wd: \n", weight_delta, "\nweights: \n", self.weights)
        # print("error: \n", error, "\n")
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

    def count_series(self, n, file):
        error = 0.0
        for j in range(n):
            file.seek(0)
            for line in file:
                fields = line.strip().split()

                input = np.array([float(fields[0]), float(fields[1]), float(fields[2])])
                expected_output = self.calc_exp_output(float(fields[3]))
                error = error + self.one_series(input, expected_output)
            # print(error, "\n")
            error = 0.0


if __name__ == '__main__':
    test_colors = 'test_colors.txt'
    training_colors = 'training_colors.txt'
    weights = np.array([[[0.26984179, 0.71716623, 0.67143705],
                        [0.7975498, 0.1757912, 0.33016793],
                        [0.7547338, 0.84341124, 0.35006583],
                        [0.02574414, 0.14325726, 0.40649936]],
                       [[0.26984179, 0.71716623, 0.67143705],
                       [0.7975498, 0.1757912, 0.33016793],
                       [0.7547338, 0.84341124, 0.35006583],
                       [0.02574414, 0.14325726, 0.40649936]]])
    nn = NeuralNetwork(weights)
    nn.read(training_colors, 50)
    print("-----------------------TEST-------------------------")
    nn.read(test_colors, 1)