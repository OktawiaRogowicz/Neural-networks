import os
import random

import numpy as np


def open_file(filename):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, filename)
    file = open(my_file)
    return file


class Word2Vec:
    def __init__(self):
        self.found_words = []
        self.matrix = []
        self.labels = []

        self.weights = []
        self.input_number = 0
        self.output_number = 0
        self.alpha = 0.01

        self.counter = 0
        self.matching = 0

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        layer = np.zeros((n, self.output_number))
        for i in range(n):
            for j in range(self.output_number):
                layer[i][j] = random.uniform(weight_min_value, weight_max_value)
        self.weights.append(layer)
        self.output_number = n

    def generate_labels_from_file(self, filename):
        file = open_file(filename)
        for line in file:
            word = line.strip("\n")
            if word == "positive":
                self.labels.append(1)
            else:
                self.labels.append(0)

    def generate_matrix_from_file(self, filename):
        self.counter = 0
        self.matching = 0
        file = open_file(filename)
        reviews = self.generate_list_from_file(file)
        print(len(reviews))
        return self.generate_multi_hot_vectors(reviews)

    def generate_list_from_file(self, file):
        reviews = list()
        words = list()
        for line in file:
            reviews.append(line.strip("\n"))
        for sentence in reviews:
            words.append(sentence.split())
        return words

    def generate_found_words(self, reviews):
        for i in range(len(reviews)):
            for j in range(len(reviews[i])):
                if reviews[i][j] not in self.found_words:
                    self.found_words.append(reviews[i][j])

    def generate_multi_hot_vectors(self, reviews):
        self.generate_found_words(reviews)
        # print(self.found_words)

        vectors = list()
        for i in range(len(reviews)):
            vector = np.zeros(len(self.found_words))
            for j in range(len(self.found_words)):
                if self.found_words[j] in reviews[i]:
                    vector[j] = 1
            vectors.append(vector)
        self.matrix = vectors
        self.input_number = len(self.matrix[0])
        self.output_number = len(self.matrix[0])

    def relu(self, weights):
        for i in range(len(weights)):
            if weights[i] < 0:
               weights[i] = 0
        return weights

    def relu_deriv(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def calc_era(self, n):
        for i in range(n):
            for j in range(len(self.matrix)):
                error = self.predict(self.matrix[j], self.labels[j])
                # print("error: ", error)

    def predict(self, input, expected_output):
        error = 0.0
        layer_1_weights = self.weights[0]
        layer_2_weights = self.weights[1]

        layer_1_values = np.dot(input, layer_1_weights.transpose())
        layer_1_values = self.relu(layer_1_values)
        layer_2_values = np.dot(layer_1_values, layer_2_weights.transpose())

        layer_2_delta = layer_2_values - expected_output
        l1v = layer_1_values.copy()
        layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * self.relu_deriv(l1v)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.weights[1] = self.weights[1] - np.dot(self.alpha, layer_2_weight_delta)
        self.weights[0] = self.weights[0] - np.dot(self.alpha, layer_1_weight_delta)

        error = error + layer_2_delta ** 2

        self.counter += 1
        if (layer_2_values > 0.5 and expected_output == 1) or\
                (layer_2_values <= 0.5 and expected_output == 0):
            self.matching += 1

        return error

    # ------------------- ZADANIE 2 -------------------

if __name__ == '__main__':
    wv = Word2Vec()
    wv.generate_matrix_from_file("reviews.txt")
    wv.generate_labels_from_file("labels.txt")
    wv.add_layer(300, -0.1, 0.1, "reLu")
    wv.add_layer(1, -0.1, 0.1, "reLu")
    wv.calc_era(2)
    print("Counter: ", wv.counter, "Matching: ", wv.matching)
    percentage = ((wv.matching * 1.0) / wv.counter) * 100.0
    print("Percentage: ", percentage, "%")
    print("-----------------------TEST-------------------------")
