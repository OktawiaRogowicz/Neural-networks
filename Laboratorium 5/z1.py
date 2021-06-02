import numpy as np


class NeuralNetwork:

    def __init__(self, input, filter):
        self.input = input
        self.filter = filter

        self.step = 1
        self.secRow = len(filter)
        self.secCol = len(filter[0])

    def count(self):
        m = self.input
        print(m, "\n")

        output = []

        for c in range(len(self.input) - (self.secCol - 1)):
            for r in range(len(self.input[0]) - (self.secRow - 1)):
                rows = []
                for i in range(3):
                    row = []
                    row.append(m[c+i][r])
                    row.append(m[c+i][r+1])
                    row.append(m[c+i][r+2])
                    rows.append(row)
                result = np.array(rows) * filter
                result = sum(sum(result))
                output.append(result)
        print(output)


if __name__ == '__main__':
    input = np.array([[1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 0],
                      [0, 1, 1, 0, 0]])
    filter = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1]])

    output_image = np.array([[4, 3, 4],
                             [2, 4, 3],
                             [2, 3, 4]])

    nn = NeuralNetwork(input, filter)
    nn.count()
