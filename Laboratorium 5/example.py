import numpy as np


class NeuralNetwork:

    def __init__(self, input, expected_output, alpha, kernel_weights, weights):
        self.input = input
        self.expected_output = expected_output
        self.alpha = alpha
        self.kernels = kernel_weights
        self.weights = weights

        self.secRow = 3
        self.secCol = 3

    def divideImage(self):
        m = self.input
        sections = []

        for c in range(len(self.input[0]) - (self.secCol - 1)):
            for r in range(len(self.input) - (self.secRow - 1)):
                section = []
                for i in range(self.secCol):
                    section.append(np.array(m[r:r+self.secRow, i]))
                section = np.concatenate(section, axis=0)
                sections.append(section)
        return sections

    def count(self):
        image_sections = self.divideImage()
        kernel_layer = np.dot(image_sections, self.kernels.transpose())

        values = np.dot(kernel_layer.flatten(), self.weights.transpose())

        layer_2_delta = values - self.expected_output
        layer_1_delta = np.dot(layer_2_delta, self.weights)
        temp = int(len(layer_1_delta) / 2)
        layer_1_delta = layer_1_delta.reshape(temp, temp)

        layer_2_weight_delta = np.dot(layer_2_delta.transpose(), kernel_layer.flatten())
        layer_1_weight_delta = np.dot(layer_1_delta.transpose(), image_sections)

        self.weights = self.weights - alpha * layer_2_weight_delta
        self.kernels = self.kernels - alpha * layer_1_weight_delta

if __name__ == '__main__':
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])
    expected_output = np.array([0, 1])
    alpha = 0.01
    kernel_weights = np.array([[0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
                               [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1]])
    weights = np.array([[0.1, -0.2, 0.1, 0.3],
                        [0.2, 0.1, 0.5, -0.3]])

    nn = NeuralNetwork(input, expected_output, alpha, kernel_weights, weights)
    nn.count()
