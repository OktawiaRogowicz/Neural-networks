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
    output = np.zeros((len(input), len(input[0])))
    if check_number_of_rows(weight, output_number) != 0 \
            or check_number_of_columns(input, weight) != 0:
        raise ValueError

    # t = weight.transpose()

    A = input
    B = weight
    C = A.dot(B.transpose())
    print(C)
    return C

    # for i in range(output_number):
    #     for j in range(len(output[i])):
    #         for k in range(len(input[i])):
    #             output[i][j] = output[i][j] + input[i][k] * weight[j][k]
    #     print(output[i])
    #
    # return output


def deep_neural_network(input, weight, output_number):
    output = np.zeros((len(weight), len(input), len(input[0])))
    # if check_number_of_rows(weight, output_number) != 0\
    #         or check_number_of_columns(input, weight) != 0\
    #         or check_number_of_layers() != 0:
    #     raise ValueError

    first_layer = input

    for i in range(len(weight)):
        A = first_layer
        B = weight[i]
        C = A.dot(B.transpose())
        first_layer = C
        print(C)

    print(C)

    # for x in range(len(weight)):
    #     if x == 0:
    #         for i in range(output_number+1):
    #             for j in range(len(output[x][i])):
    #                 for k in range(len(input[i])):
    #                     output[x][i][j] = output[x][i][j] + input[i][k] * weight[x][j][k]
    #     else:
    #         for i in range(output_number+1):
    #             for j in range(len(output[x][i])):
    #                 for k in range(len(input[i])):
    #                     output[x][i][j] = output[x][i][j] + output[x-1][i][k] * weight[x][j][k]
    # print(output)

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
