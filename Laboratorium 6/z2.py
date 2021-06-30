import random as rand
import numpy as np


class Population:
    def __init__(self, population_n, genes_n):
        self.population_n = population_n
        self.population = []
        self.new_population = []
        self.genes_n = genes_n
        self.fitness_sum = 0.0

    def __generate_gene(self):
        return list(np.random.randint(low=2, size=self.genes_n))

    def generate_population(self):
        for i in range(self.population_n):
            result = self.__generate_gene()
            self.population.append(result)
            self.fitness_sum += self.__fitness_function(result)
        self.__segregate_population()

    def __fitness_function(self, chromosome):

        a = int("".join(str(i) for i in chromosome[:4]), 2)
        b = int("".join(str(i) for i in chromosome[4:]), 2)

        # result = abs(((2 * (a ** 2) + b) / 33.0) - 1)
        result = 1 / ((abs((2 * (a ** 2) + b) - 33.0)) + 1)
        return result

    def print_population(self):
        for chromosome in self.population:
            print(self.__fitness_function(chromosome))

    def __segregate_population(self):
        print("PRZED SEGREGACJA: ", self.population)
        self.population.sort(key=self.__fitness_function, reverse=True)
        print("PO SEGREGACJI: ", self.population, "\n")
        # self.print_population()

    def __roulette(self):
        pick = rand.uniform(0, self.fitness_sum)
        current = 0
        for chromosome in self.population:
            current += self.__fitness_function(chromosome)
            if current > pick:
                return chromosome

    def roulette_wheel_selection(self):
        self.new_population.clear()
        for i in range(self.population_n):
            self.new_population.append(self.__roulette())

    def create_children(self):
        indexes = list(np.random.randint(low=self.genes_n, size=self.population_n // 2))
        new_copy = self.new_population.copy()

        for i in indexes:
            chromosome1 = self.new_population[i]
            index2 = [j for j in indexes if j != i]
            index2 = rand.choice(index2)
            chromosome2 = self.new_population[index2]

            child_chromosome1 = []
            result = rand.sample(range(0, self.genes_n - 1), 1)
            result = result[0]

            child_chromosome1.extend(chromosome1[:result])
            child_chromosome1.extend(chromosome2[result:])

            self.__mutation_of_one_chromosome(child_chromosome1)

            which_parent = rand.random()
            if which_parent > 0.5:
                new_copy[index2] = child_chromosome1
            else:
                new_copy[i] = child_chromosome1

        self.population = new_copy
        self.__segregate_population()

    def __mutation_of_one_chromosome(self, chromosome):
        if rand.randint(0, 100) <= 10:
            index = rand.randint(0, self.genes_n - 1)
            if chromosome == 0:
                chromosome = 1
            else:
                chromosome = 0


if __name__ == '__main__':
    p = Population(10, 8)
    p.generate_population()

    for i in range(10):
        p.roulette_wheel_selection()
        p.create_children()
