import os

import numpy as np


class Word2Vec:
    def __init__(self, file, n):
        self.found_words = []
        self.matrix = []
        self.file = file
        self.n = n

    def generate_matrix_from_file(self):
        reviews = self.generate_list_from_file()
        return self.generate_multi_hot_vectors(reviews)

    def generate_list_from_file(self):
        reviews = list()
        words = list()
        for line in self.file:
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
        print(self.found_words)

        vectors = list()
        for i in range(len(reviews)):
            vector = np.zeros(len(self.found_words))
            for j in range(len(self.found_words)):
                if self.found_words[j] in reviews[i]:
                    vector[j] = 1
            vectors.append(vector)
        print(vectors)


if __name__ == '__main__':
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, "text.txt")
    file = open(my_file)

    wv = Word2Vec(file, 1)
    wv.generate_matrix_from_file()
