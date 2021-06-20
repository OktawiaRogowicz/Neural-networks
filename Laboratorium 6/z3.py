import random as rand
import numpy as np


class Population:
    def __init__(self, objects, population_n):
        self.objects = objects
        self.population_n = population_n
        self.population = []
        self.new_population = []
        self.genes_n = len(objects)
        self.fitness_sum = 0.0

    def print_population(self):
        for chromosome in self.population:
            print(self.__fitness_function(chromosome))

    def __generate_gene(self):
        return list(np.random.randint(low=2, size=self.genes_n))

    def generate_population(self):
        for i in range(self.population_n):
            result = self.__generate_gene()
            self.population.append(result)
        self.print_population()
        self.__update_fitness_sum()

    def __update_fitness_sum(self):
        sum = 0
        for chromosome in self.population:
            sum += self.__fitness_function(chromosome)
        self.fitness_sum = sum

    def __segregate_population(self):
        print("PRZED SEGREGACJA: ", [self.__fitness_function(chromosome) for chromosome in self.population])
        self.population.sort(key=self.__fitness_function, reverse=True)
        print("PO SEGREGACJI: ", [self.__fitness_function(chromosome) for chromosome in self.population], "\n")

    def __fitness_function(self, chromosome):
        weight = 0
        value = 0

        for i in range(len(chromosome)):
            gene = chromosome[i]
            weight += gene * self.objects[i][0]
            value += gene * self.objects[i][1]

        if weight > 35:
            return 0
        else:
            return value

    def __roulette(self):
        pick = rand.uniform(0, self.fitness_sum)
        current = 0
        for chromosome in self.population:
            current += self.__fitness_function(chromosome)
            if current > pick:
                return chromosome

    def roulette_wheel_selection(self):
        self.new_population.clear()
        for i in range(self.population_n - 2):
            self.new_population.append(self.__roulette())

    def create_children(self):
        indexes = list(np.random.randint(low= self.population_n - 2, size=self.population_n - 2))
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

        new_copy.append(self.population[0])
        new_copy.append(self.population[1])
        self.population = new_copy
        self.__update_fitness_sum()
        self.__segregate_population()

    def __mutation_of_one_chromosome(self, chromosome):
        for i in range(len(chromosome)):
            if rand.randint(0, 100) <= 5:
                if chromosome[i] == 0:
                    chromosome[i] = 1
                else:
                    chromosome[i] = 0


if __name__ == '__main__':
    objects = np.array([[3, 266],
                        [13, 442],
                        [10, 671],
                        [9, 526],
                        [7, 388],
                        [1, 245],
                        [8, 210],
                        [8, 145],
                        [2, 126],
                        [9, 322]])

    p = Population(objects, 8)
    p.generate_population()

    for i in range(50):
        p.roulette_wheel_selection()
        p.create_children()
