import random as rand
import numpy as np


class Population:
    def __init__(self, population_n, genes_n):
        self.population_n = population_n
        self.population = []
        self.genes_n = genes_n

    def __generate_gene(self):
        return list(np.random.randint(low=2, size=self.genes_n))

    def generate_population(self):
        for i in range(self.population_n):
            result = self.__generate_gene()
            self.population.append(result)
        self.__segregate_population()

    def __fitness_function(self, chromosome):
        return chromosome.count(1)

    def __segregate_population(self):
        print("PRZED SEGREGACJA: ", self.population)
        self.population.sort(key=self.__fitness_function, reverse=True)
        print("PO SEGREGACJI: ", self.population, "\n")

    def create_two_children_chromosomes(self):
        chromosome1 = self.population[0]
        chromosome2 = self.population[1]

        child_chromosome1 = []
        child_chromosome2 = []
        result = rand.sample(range(0, self.genes_n - 1), 1)
        result = result[0]

        child_chromosome1.extend(chromosome1[:result])
        child_chromosome1.extend(chromosome2[result:])

        child_chromosome2.extend(chromosome2[:result])
        child_chromosome2.extend(chromosome1[result:])

        self.population[self.population_n - 1] = child_chromosome1
        self.population[self.population_n - 2] = child_chromosome2
        self.__segregate_population()

    def mutate(self):
        self.__mutation_of_one_chromosome(self.population[0])
        self.__mutation_of_one_chromosome(self.population[1])

    def __mutation_of_one_chromosome(self, chromosome):
        if rand.randint(0, self.population_n) >= 6:
            index = rand.randint(0, self.genes_n - 1)
            if chromosome[index] == 0:
                chromosome[index] = 1
            else:
                chromosome[index] = 0


if __name__ == '__main__':
    p = Population(10, 10)
    p.generate_population()

    for i in range(10):
        p.create_two_children_chromosomes()
        p.mutate()
