import os


# RED [1], GREEN [2], BLUE [3], YELLOW [4]
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.alpha = 0.01

    def read(self, filename, n):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, filename)
        file = open(my_file)
        self.count_series(n, file)
        file.close()

    def one_series(self, n, input, weights, expected_output):
        A = input
        B = weights
        C = A.dot(B.transpose())
        prediction = C
        # print("C: \n", C, "\n\n")

        error = pow(prediction - expected_output, 2)
        delta = prediction - expected_output
        # print("delta: \n", delta)
        weight_delta = np.outer(delta, input[n])
        weights = weights - weight_delta * self.alpha
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

    def calc_weights(self, input, expected_output):
        print("Input: \n", input)
        print("Exp out: \n", expected_output)
        weights = np.array([input * expected_output[0],
                           input * expected_output[1],
                           input * expected_output[2],
                           input * expected_output[3]])
        print("WEIGHTS: \n", weights)
        return weights

    def count_series(self, n, file):
        error = 0.0
        for j in range(n):
            for line in file:
                fields = line.strip().split()

                input = np.array([float(fields[0]), float(fields[1]), float(fields[2])])
                expected_output = self.calc_exp_output(float(fields[3]))
                weights = self.calc_weights(input, expected_output)

                # error = error + self.one_series(input, weights, expected_output)
            # print(error, "\n")
            # print(error[0] + error[1] + error[2])
            error = 0.0


if __name__ == '__main__':
    test_colors = 'test_colors.txt'
    training_colors = 'training_colors.txt'
    nn = NeuralNetwork()
    nn.read(training_colors, 1)

