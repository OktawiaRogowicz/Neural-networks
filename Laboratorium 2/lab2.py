import numpy as np


def z1(input, weight, goal, alpha, n):
    for i in range(n):
        prediction = weight * input

        error = pow(prediction - goal, 2)
        delta = prediction - goal
        weight_delta = input * delta
        weight = weight - weight_delta * alpha

        print("Iteracja ", i + 1, ": ", "{:.15f}".format(prediction))
        print("    error: ", "{:.15f}".format(error))


def z2(input, weights, goal, alpha):
    error = 0.0
    for j in range(100):
        for i in range(len(input)):
            error = error + fun(i, input, weights, goal, alpha)
        print("{:.15f}".format(error))
        error = 0.0


def fun(n, input, weights, goal, alpha):
    weight_delta = np.zeros(len(weights))

    A = input[n]
    B = weights[0]
    C = A.dot(B.transpose())
    prediction = C
    error = pow(prediction - goal[n][0], 2)
    delta = prediction - goal[n][0]
    print("delta: ", delta)
    weight_delta = input[n] * delta
    weights[0] = weights[0] - weight_delta * alpha
    print("wd: ", weight_delta, "\nweights: ", weights)
    print("error: ", "{:.15f}".format(error), "\n")

    return error


def z3(input, weights, expected_output, alpha):
    error = 0.0
    for j in range(50):
        for i in range(len(input)):
            error = error + fun3(i, input, weights, expected_output, alpha)
        print(error, "\n")
        error = 0.0


def fun3(n, input, weights, expected_output, alpha):
    e = 0.0
    A = input[n]
    B = weights[0]
    C = A.dot(B.transpose())
    prediction = C
    # print("C: \n", C, "\n\n")

    error = pow(prediction - expected_output[n], 2)
    delta = prediction - expected_output[n]
    # print("delta: ", delta)
    weight_delta = np.outer(input[n], delta)
    weights[0] = weights - weight_delta * alpha
    # print("wd: ", weight_delta, "\nweights: ", weights)
    print(weights)
    # print("error: ", error, "\n")
    return error


if __name__ == '__main__':
    # z1(2, 0.5, 0.8, 0.1, 20)
    # z1(0.1, 0.5, 0.8, 1, 20)
    # ZADANIE 2
    weights = np.array([[0.1, 0.2, -0.1],
                       [0.1, 0.2, -0.1],
                       [0.1, 0.2, -0.1],
                       [0.1, 0.2, -0.1]])
    input = np.array([[8.5, 0.65, 1.2],
                      [9.5, 0.8, 1.3],
                      [9.9, 0.8, 0.5],
                      [9.0, 0.9, 1.0]])
    expected_output = np.array([[1], [1], [0], [1]])
    alpha = 0.01
    z2(input, weights, expected_output, alpha)

    # ZADANIE 3
    weights3 = np.array([[[0.1, 0.1, -0.3],
                         [0.1, 0.2, 0.0],
                         [0.0, 1.3, 0.1]],
                        [[0.1, 0.1, -0.3],
                         [0.1, 0.2, 0.0],
                         [0.0, 1.3, 0.1]]])
    input3 = input
    expected_output3 = [[0.1, 1, 0.1],
                        [0, 1, 0],
                        [0, 0, 0.1],
                        [0.1, 1, 0.2]]
    alpha3 = alpha
    z3(input3, weights3, expected_output3, alpha3)
