import numpy as np


def relu(weights):
    weights[weights < 0.0] = 0
    return weights


def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class NeuralNetwork:

    def __init__(self, input_number, alpha):
        self.weights = 0
        self.kernels = 0
        self.input_number = input_number
        self.output_number = input_number
        self.alpha = alpha

        self.kernelCounter = 0

        self.secRow = 3
        self.secCol = 3

    def add_kernel_filters(self, n, cols, rows, weight_min_value, weight_max_value):
        self.kernelCounter = n
        layer = np.random.uniform(weight_min_value, weight_max_value, size=(n, cols * rows))
        self.kernels = layer

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        layer = np.random.uniform(weight_min_value, weight_max_value, size=(n, self.output_number * self.kernelCounter))
        self.weights = layer
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

    def file_to_vectors(self, n, q, file_name_images, file_name_labels):
        self.matching = 0
        self.counter = 0

        [baI, quantity, rows, columns] = self.read_images(file_name_images)
        baL = self.read_labels(file_name_labels)

        for x in range(n):
            for i in range(q):
                vector = self.one_picture_to_vector(baI, columns, rows, i)
                label = self.one_label_to_int(baL, i)
                expected_output = self.calc_exp_output(label)
                print(x, i)
                error = self.predict(np.asarray(vector), np.asarray(expected_output))

    def one_picture_to_vector(self, ba, columns, rows, n):
        vector = np.array(np.zeros([columns, rows]))
        for i in range(columns):
            for j in range(rows):
                x = columns * rows * n + 16
                vector[i][j] = ba[x + i * columns + j]
                vector[i][j] = vector[i][j] / 255.0
        return vector

    def one_label_to_int(self, ba, n):
        return ba[n + 8]

    def calc_exp_output(self, expected_output):
        expected = np.array(np.zeros(10))
        expected[expected_output] = 1.0

        return expected

    def divide_image(self, input):
        sections = []

        for i in range(len(input) - (self.secRow - 1)):
            for j in range(len(input[0]) - (self.secCol - 1)):
                rows = []
                for a in range(3):
                    row = []
                    row.append(input[i+a][j])
                    row.append(input[i+a][j+1])
                    row.append(input[i+a][j+2])
                    rows.append(row)
                section = np.concatenate(rows, axis=0)
                sections.append(section)
        return sections

    def predict(self, input, expected_output):
        image_sections = self.divide_image(input)
        kernel_layer = np.dot(image_sections, self.kernels.transpose())

        kernel_layer_flat = kernel_layer.flatten()
        values = np.dot(kernel_layer_flat, self.weights.transpose())

        layer_2_delta = values - expected_output
        layer_1_delta = np.dot(layer_2_delta, self.weights)
        layer_1_delta *= relu_deriv(kernel_layer)

        temp1 = int(len(kernel_layer))
        temp2 = int(len(kernel_layer[0]))
        layer_1_delta = layer_1_delta.reshape(temp1, temp2)

        l2t = layer_2_delta.reshape((1, -1)).transpose()
        kt = kernel_layer.flatten().reshape((1, -1))

        layer_2_weight_delta = np.dot(l2t, kt)
        layer_1_weight_delta = np.dot(layer_1_delta.transpose(), image_sections)

        self.weights = self.weights - self.alpha * layer_2_weight_delta
        self.kernels = self.kernels - self.alpha * layer_1_weight_delta
        return 0


if __name__ == '__main__':
    nn = NeuralNetwork(676, 0.005)

    # Wersja 1:
    nn.add_kernel_filters(16, 3, 3, -0.01, 0.01)
    nn.add_layer(10, -0.1, 0.1, "reLu")
    nn.file_to_vectors(50, 1000, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)
    nn.file_to_vectors(1, 10000, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print("Counter: ", nn.counter, " | Matching: ", nn.matching)