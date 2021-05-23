import random
import numpy as np

class NeuralNetwork:
    def __init__(self, input_number):
        self.weights = []
        self.input_number = input_number
        self.output_number = input_number
        self.alpha = 0.01

        self.input = []
        self.expected_output = []

        self.counter = 0
        self.matching =0

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        # dodaje warstwe n neuronow do sieci
        # + losowanie wag w zakresie

        layer = np.zeros((n, self.output_number))
        for i in range(n):
            for j in range(self.output_number):
                layer[i][j] = random.uniform(weight_min_value, weight_max_value)
        self.weights.append(layer)
        self.output_number = n

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

    def file_to_vectors(self, file_name_images, file_name_labels):
        self.matching = 0
        self.counter = 0

        [baI, quantity, rows, columns] = self.read_images(file_name_images)
        baL = self.read_labels(file_name_labels)

        for i in range(quantity):
            vector = self.one_picture_to_vector(baI, columns, rows, i)
            label = self.one_label_to_int(baL, i)
            expected_output = self.calc_exp_output(label)
            error = self.predict(np.asarray(vector), np.asarray(expected_output))

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

    def relu(self, weights):
        for i in range(len(weights)):
            if weights[i] < 0:
               weights[i] = 0
        return weights

    def relu_deriv(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def predict(self, input, expected_output):
        error = 0.0
        layer_1_weights = self.weights[0]
        layer_2_weights = self.weights[1]

        layer_1_values = np.dot(input, layer_1_weights.transpose())
        layer_1_values = self.relu(layer_1_values)
        layer_2_values = np.dot(layer_1_values, layer_2_weights.transpose())

        index = np.argmax(layer_2_values)
        index2 = np.argmax(expected_output)

        layer_2_delta = layer_2_values - expected_output
        l1v = layer_1_values.copy()
        layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * self.relu_deriv(l1v)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.weights[1] = self.weights[1] - np.dot(self.alpha, layer_2_weight_delta)
        self.weights[0] = self.weights[0] - np.dot(self.alpha, layer_1_weight_delta)

        error = error + layer_2_delta ** 2

        if index == index2:
            self.matching += 1
        self.counter += 1

        return error


if __name__ == '__main__':
    nn = NeuralNetwork(784)
    nn.add_layer(40, -0.1, 0.1, "reLu")
    nn.add_layer(10, -0.1, 0.1, "reLu")
    nn.file_to_vectors("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
    nn.file_to_vectors("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)