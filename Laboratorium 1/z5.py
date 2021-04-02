import random
import numpy as np


# def deep_neural_network(self, weight):
#     output = np.zeros((len(weight), len(self.input), len(self.input[0])))
#
#     for x in range(len(weight)):
#         if x == 0:
#             for i in range(output_number + 1):
#                 for j in range(len(output[x][i])):
#                     for k in range(len(self.input[i])):
#                         output[x][i][j] = output[x][i][j] + self.input[i][k] * weight[x][j][k]
#         else:
#             for i in range(self.output_number + 1):
#                 for j in range(len(output[x][i])):
#                     for k in range(len(self.input[i])):
#                         output[x][i][j] = output[x][i][j] + output[x - 1][i][k] * weight[x][j][k]
#     print(output)

class NeuralNetwork:
    def __init__(self, input_number):
        self.layers = []
        self.input_number = input_number
        self.output_number = input_number

    def add_layer(self, n, weight_min_value, weight_max_value):
        # dodaje warstwe n neuronow do sieci
        # + losowanie wag w zakresie

        layer = np.zeros((n, self.output_number))
        for i in range(n):
            for j in range(self.output_number):
                layer[i][j] = random.uniform(weight_min_value, weight_max_value)
        self.layers.append(layer)
        self.output_number = n
        print("layer: ", n, "\n", layer)

    def predict(self, input):

        first_layer = input

        for i in range(len(self.layers)):
            print("\nPREDICT FOR: ", first_layer)
            A = first_layer
            B = self.layers[i]
            print("Pierwsza macierz: \n", A, "\nDruga macierz: \n", B)
            C = A.dot(B.transpose())
            print("\nWynik: \n", C)
            first_layer = C
        return


if __name__ == '__main__':
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])

    # NN = NeuralNetwork(3) // liczba wejść
    # NN.add_layer(5) // dodaje warstw z piecioma neuronami
    # NN.add_layer(3) // dodaje warstwę z trzema neuronami
    # NN.predict([1, 0.85, 0.2]) // wykonuje obliczenia na dwóch warstwach wcześniej dodanych

    nn = NeuralNetwork(3)
    nn.add_layer(5, -0.3, 0.3)
    nn.add_layer(3, -0.3, 0.3)
    input = np.array([1, 0.85, 0.2])
    nn.predict(input)
