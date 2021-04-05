# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def z1(input, weight):
    output = weight * input
    return output


def z2(input, weight):
    output = 0.0

    for i in range(len(input)):
        output = output + (input[i] * weight[i])
    return output


def check_number_of_rows(weight, output_number):
    for i in range(len(weight)):
        if len(weight[i]) != output_number:
            raise ValueError
    return 0


def check_number_of_columns(input, weight):
    for i in range(len(input)):
        print(len(weight), len(input[i]))
        if len(weight) != len(input[i]):
            raise ValueError
    return 0


def neural_network(input, weight, output_number):
    if check_number_of_rows(weight, output_number) != 0 \
            or check_number_of_columns(input, weight) != 0:
        raise ValueError

    A = input
    B = weight
    C = A.dot(B.transpose())
    print(C)
    return C


def deep_neural_network(input, weight, output_number):
    first_layer = input
    for i in range(len(weight)):
        A = first_layer
        B = weight[i]
        C = A.dot(B.transpose())
        first_layer = C
        print(C)

    print(C)


if __name__ == '__main__':
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])

    weight = np.array([[0.1, 0.1, -0.3],
                       [0.1, 0.2, 0.0],
                       [0.0, 1.3, 0.1]])

    weight2 = np.array([[[0.1, 0.2, -0.1],
                         [-0.1, 0.1, 0.9],
                         [0.1, 0.4, 0.1]],
                        [[0.3, 1.1, -0.3],
                         [0.1, 0.2, 0.0],
                         [0.0, 1.3, 0.1]]])
    number = 3
    # neural_network(input, weight, number)
    deep_neural_network(input, weight2, number)
