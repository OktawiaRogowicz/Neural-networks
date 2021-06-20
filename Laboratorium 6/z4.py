import random as rand
import numpy as np


class Population:
    def __init__(self, objects, population_n, elitarism):
        self.objects = objects
        self.population_n = population_n
        self.population = []
        self.new_population = []
        self.genes_n = len(objects)
        self.fitness_sum = 0.0
        self.elite_items = int(population_n / 100 * elitarism)

    def __generate_gene(self):
        return rand.sample(range(len(objects)), len(objects))

    def generate_population(self):
        for i in range(self.population_n):
            result = self.__generate_gene()
            self.population.append(result)
        self.__update_fitness_sum()

    def __update_fitness_sum(self):
        sum = 0
        for chromosome in self.population:
            sum += self.__fitness_function(chromosome)
        self.fitness_sum = sum

    def __segregate_population(self):
        print("PRZED SEGREGACJA: ", [self.__fitness_function(chromosome) for chromosome in self.population])
        self.population.sort(key=self.__fitness_function, reverse=False)
        print("PO SEGREGACJI: ", [self.__fitness_function(chromosome) for chromosome in self.population], "\n")

    def __fitness_function(self, chromosome):
        sum = 0
        for i in range(len(chromosome) - 1):
            result = ((
                        (self.objects[chromosome[i+1]][0] - self.objects[chromosome[i]][1]) ** 2 +
                        (self.objects[chromosome[i+1]][1] - self.objects[chromosome[i]][1]) ** 2
                        )
                      ** 0.5)
            sum += result
        return sum

    def __roulette(self):
        pick = rand.uniform(0, self.fitness_sum)
        current = 0
        for chromosome in self.population:
            current += self.__fitness_function(chromosome)
            if current > pick:
                return chromosome

    def roulette_wheel_selection(self):
        self.new_population.clear()
        for i in range(self.population_n - self.elite_items):
            self.new_population.append(self.__roulette())

    def create_children(self):
        indexes = list(
            np.random.randint(low=self.population_n - self.elite_items, size=self.population_n - self.elite_items))
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

        for j in range(self.elite_items):
            new_copy.append(self.population[j])

        self.population = new_copy
        self.__update_fitness_sum()
        self.__segregate_population()

    def __mutation_of_one_chromosome(self, chromosome):
        for i in range(len(chromosome)):
            if rand.randint(0, 100) <= 1:
                indexes = rand.sample(range(self.genes_n), 2)
                temp = chromosome[indexes[0]]
                chromosome[indexes[0]] = chromosome[indexes[1]]
                chromosome[indexes[1]] = temp


if __name__ == '__main__':
    objects = np.array([[119, 38],
                        [37, 38],
                        [197, 55],
                        [85, 165],
                        [12, 50],
                        [100, 53],
                        [81, 142],
                        [121, 137],
                        [85, 145],
                        [80, 197],
                        [91, 176],
                        [106, 55],
                        [123, 57],
                        [40, 81],
                        [78, 135],
                        [190, 46],
                        [187, 40],
                        [37, 107],
                        [17, 11],
                        [67, 56],
                        [78, 133],
                        [87, 23],
                        [184, 197],
                        [111, 12],
                        [66, 178]])

    p = Population(objects, 100, 20)
    p.generate_population()

    for i in range(100):
        p.roulette_wheel_selection()
        p.create_children()
    print(p.population)
