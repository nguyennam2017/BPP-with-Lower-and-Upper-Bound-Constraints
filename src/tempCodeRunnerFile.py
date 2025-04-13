from random import randint, sample, random, choice
import numpy as np

class GeneticAlgorithmVRP:
    def __init__(self, distance_matrix, demands, vehicle_capacity, population_size=50, generations=100, mutation_rate=0.1):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.N = len(demands)
        self.initialization_methods = [self.greedy_initializer, self.random_initializers, self.balanced_initializer]

    def greedy_initializer(self):
        population = []
        for _ in range(self.population_size):
            route = []
            vehicle_load = 0
            for i in range(self.N):
                if vehicle_load + self.demands[i] <= self.vehicle_capacity:
                    route.append(i)
                    vehicle_load += self.demands[i]
                else:
                    route.append(-1)
                    route.append(i)
                    vehicle_load = self.demands[i]
            population.append(route)
        return population

    def random_initializers(self):
        population = []
        for _ in range(self.population_size):
            orders = list(range(self.N))
            route = []
            vehicle_load = 0
            while orders:
                i = orders.pop(randint(0, len(orders) - 1))
                if vehicle_load + self.demands[i] <= self.vehicle_capacity:
                    route.append(i)
                    vehicle_load += self.demands[i]
                else:
                    route.append(-1)
                    vehicle_load = self.demands[i]
                    route.append(i)
            population.append(route)
        return population

    def balanced_initializer(self):
        population = []
        for _ in range(self.population_size):
            sorted_orders = sorted(range(self.N), key=lambda x: self.demands[x], reverse=True)
            route = []
            vehicles = [[]]
            loads = [0]
            for order in sorted_orders:
                placed = False
                for i in range(len(vehicles)):
                    if loads[i] + self.demands[order] <= self.vehicle_capacity:
                        vehicles[i].append(order)
                        loads[i] += self.demands[order]
                        placed = True
                        break
                if not placed:
                    vehicles.append([order])
                    loads.append(self.demands[order])
            for i, v in enumerate(vehicles):
                if i > 0:
                    route.append(-1)
                route += v
            population.append(route)
        return population

    def initialize_population(self):
        method = choice(self.initialization_methods)
        return method()

    def fitness(self, individual, penalty_weight=100):
        total_distance = 0
        capacity = 0
        last_node = 0
        penalty = 0
        for node in individual:
            if node == -1:
                total_distance += self.distance_matrix[last_node][0]
                last_node = 0
                capacity = 0
            else:
                capacity += self.demands[node]
                if capacity > self.vehicle_capacity:
                    penalty += penalty_weight * (capacity - self.vehicle_capacity)
                total_distance += self.distance_matrix[last_node][node]
                last_node = node
        total_distance += self.distance_matrix[last_node][0]
        return total_distance + penalty

    def tournament_selection(self, population, fitness_values, tournament_size=3):
        selected = sample(list(zip(population, fitness_values)), tournament_size)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def crossover(self, p1, p2):
        point1 = randint(0, len(p1) - 1)
        point2 = randint(point1, len(p1))
        child1 = p1[:]
        child2 = p2[:]
        for i in range(point1, point2):
            if p1[i] != -1 and p2[i] != -1:
                child1[i], child2[i] = p2[i], p1[i]
        return child1, child2

    def mutate(self, individual):
        mutation_type = choice(['reassign', 'swap', 'remove', 'neighbor'])
        if mutation_type == 'swap':
            i, j = sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        elif mutation_type == 'remove':
            if len(individual) > 2:
                individual.pop(randint(0, len(individual) - 1))
        elif mutation_type == 'reassign':
            positions = [i for i, x in enumerate(individual) if x != -1]
            if len(positions) > 1:
                i = choice(positions)
                val = individual.pop(i)
                insert_pos = randint(0, len(individual))
                individual.insert(insert_pos, val)
        elif mutation_type == 'neighbor':
            positions = [i for i, x in enumerate(individual) if x != -1]
            if len(positions) > 1:
                i = choice(positions)
                neighbor = (individual[i] + 1) % self.N
                individual[i] = neighbor

    def evolve(self):
        population = self.initialize_population()
        best_individual = None
        best_fitness = float('inf')
        no_improvement_count = 0
        max_no_improvement = 50

        for gen in range(self.generations):
            fitness_values = [self.fitness(ind) for ind in population]
            new_population = []

            # Elitism: keep best individual
            best_idx = np.argmin(fitness_values)
            new_population.append(population[best_idx])
            current_best_fitness = fitness_values[best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[best_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                child1, child2 = self.crossover(parent1, parent2)

                if random() < self.mutation_rate:
                    self.mutate(child1)
                if random() < self.mutation_rate:
                    self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            if no_improvement_count >= max_no_improvement:
                print(f"Stopping early at generation {gen} due to no improvement.")
                break

        return best_individual, best_fitness

