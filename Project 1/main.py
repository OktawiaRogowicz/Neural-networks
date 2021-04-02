import os
import numpy as np
from scipy import spatial
from heapq import nlargest


class WordVectors:
    def __init__(self, file, n):
        self.found_words = []
        self.matrix = []
        self.file = file
        self.n = n

    def generate_matrix_from_file(self):
        sentences = self.generate_list_from_file()
        return self.generate_matrix(sentences)

    def generate_list_from_file(self):
        sentences = list()
        words = list()
        for line in self.file:
            sentences.append(line.strip("\n"))
        for sentence in sentences:
            words.append(sentence.split())
        return words

    def generate_found_words(self, sentences):
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j] not in self.found_words:
                    self.found_words.append(sentences[i][j])

    def generate_matrix(self, sentences):
        self.generate_found_words(sentences)
        print(self.found_words, self.file)
        self.matrix = np.zeros((len(self.found_words), len(self.found_words)))

        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                for k in range(self.n):
                    if j - (k+1) >= 0:
                        index_found = self.found_words.index(sentences[i][j-(k+1)])
                        index_current = self.found_words.index(sentences[i][j])
                        self.matrix[index_current][index_found] = self.matrix[index_current][index_found] + 1

                for k in range(self.n):
                    if j + (k + 1) < len(sentences[i]):
                        index_found = self.found_words.index(sentences[i][j+(k+1)])
                        index_current = self.found_words.index(sentences[i][j])
                        self.matrix[index_current][index_found] = self.matrix[index_current][index_found] + 1
        return self.matrix

    def cosine_similarity(self, word):
        if word not in self.found_words:
            raise Exception("Word not found in the file")

        s = []
        index = self.found_words.index(word)
        p = self.matrix[index]

        for i in range(len(self.found_words)):
            q = self.matrix[i]
            s.append(1 - spatial.distance.cosine(p, q))

        res = nlargest(10, s)
        print(res)
        return s


if __name__ == '__main__':
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, "sentences2.txt")
    file = open(my_file)

    wv = WordVectors(file, 1)
    matrix = wv.generate_matrix_from_file()
    wv.cosine_similarity("ecuador")
