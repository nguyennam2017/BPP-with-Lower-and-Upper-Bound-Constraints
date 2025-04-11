import random
import numpy as np
n, k = map(int, input().split())
d = []
c = []
c1 = []
c2 = []
for i in range(n):
    _d, _c = map(int, input().split())
    d.append(_d)
    c.append(_c)

for i in range(k):
    _c1, _c2 = map(int, input().split())
    c1.append(_c1)
    c2.append(_c2)
def fitness(individual):
    quantity = [0] * k
    cost = [0] * k

    for i in range(n):
        trunk = individual[i]
        if trunk != -1:
            quantity[trunk] += d[i]
            cost[trunk] += c[i]
    
    total_cost = 0
    penaty = 0

    for i in range(k):
        if c1[i] <= quantity[i] <= c2[i]:
            total_cost += cost[i]
        elif quantity[i] < c1[i]:
            penaty += 100 * (c1[i] - quantity[i])
        else:
            penaty += 100 * (quantity[i] - c2[i])
    
    return total_cost - penaty
def greedy_initializer():
    individual = [-1] * n
    ratios = [(i,c[i]/d[i]) for i in range(n)]

    ratios.sort(key=lambda x: x[1], reverse=True)

    vehicle_load = [0] * k
    for order_idx, _ in ratios:
        best_vehicle = -1
        best_remaning = float('inf')

        for v in range(k):
            new_load = vehicle_load[v] + d[order_idx]
            # Kiểm tra nếu thêm đơn hàng vẫn nằm trong giới hạn tải
            if c1[v] <= new_load <= c2[v]:
                # Tìm xe có dung lượng còn lại ít nhất sau khi gán
                remaining = c2[v] - new_load
                if remaining < best_remaning:
                    best_remaining = remaining
                    best_vehicle = v
        
        if best_vehicle != -1:
            individual[order_idx] = best_vehicle
            vehicle_load[best_vehicle] += d[order_idx]

    
    return individual
        

    
def random_initializers():
    return [random.randint(-1,k-1) for _ in range(n)]

def initializers_population(pop_size, greedy_ratio = 0.4):
    population = []
    greedy_count = int(pop_size * greedy_ratio)
    for _ in range(greedy_count):
        population.append(greedy_initializer())

    for _ in range(pop_size - greedy_count):
        population.append(random_initializers())

    return population
def crosserver(p1, p2):
    if len(p1) < 2:
        return p1.copy(), p2.copy()

    point1 = random.randint(0, len(p1) - 2)
    point2 = random.randint(point1 + 1, len(p1) - 1)

    child1 = p1.copy()
    child2 = p2.copy()

    for i in range(point1, point2):
        # Chỉ hoán đổi nếu cả hai vị trí không phải -1
        if p1[i] != -1 and p2[i] != -1:
            child1[i], child2[i] = p2[i], p1[i]

    return child1, child2


def mutate(individual, mutation_rate=0.1):
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutation_type = random.choice(['reassign', 'swap', 'remove', 'neighbor'])

            if mutation_type == 'reassign':
                mutated[i] = random.randint(-1, k - 1)

            elif mutation_type == 'remove':
                mutated[i] = -1

            elif mutation_type == 'neighbor':
                if mutated[i] != -1:
                    shift = random.choice([-1, 1])
                    new_val = mutated[i] + shift
                    if 0 <= new_val < k:
                        mutated[i] = new_val

            elif mutation_type == 'swap':
                j = random.randint(0, len(mutated) - 1)
                mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated

def tournament_selection(population, fitness_values, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)

    best_idx = tournament_indices[0]

    for i in tournament_indices:
        if fitness_values[i] > fitness_values[best_idx]:
            best_idx = i
    
    return population[best_idx]
def GA(pop_size = 100, generations = 1000, crosserver_rate=0.7, mutaion_rate=0.2, tourrnaments_size=3, greedy_ratio=0.4, elitism=0.2):
    population = initializers_population(pop_size=pop_size, greedy_ratio=greedy_ratio)

    best_individual = None
    best_fitness = float('-inf')

    for gen in range(generations):
        fitness_values = [fitness(ind) for ind in population]

        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_values[i] for i in sorted_indices]

        if sorted_fitness[0] > best_fitness:
            best_fitness = sorted_fitness[0]
            best_individual = sorted_population[0].copy()

        new_populatiom = []

        elitism_count = int(pop_size * elitism)
        new_populatiom.extend(ind.copy() for ind in sorted_population[:elitism_count])


        while len(new_populatiom) < pop_size:
            p1 = tournament_selection(population, fitness_values, tourrnaments_size)
            p2 = tournament_selection(population, fitness_values, tourrnaments_size)


            child1, child2 = crosserver(p1, p2)

            new_populatiom.append(child1)
            if len(new_populatiom) < pop_size:
                new_populatiom.append(child2)
    
        population = new_populatiom
    
    final_fitness_value = [fitness(ind) for ind in population]
    final_best_idx = final_fitness_value.index(max(final_fitness_value))

    if final_fitness_value[final_best_idx] > best_fitness:
        best_individual = population[final_best_idx].copy()
        best_fitness = final_fitness_value[final_best_idx]
    
    return best_individual, best_fitness
if __name__ == "__main__":
    pop_size = 100
    generations = 20000
    crossover_rate = 0.7
    mutation_rate = 0.2
    tournament_size = 3
    greedy_ratio = 0.4  # 40% cá thể tham lam, 60% ngẫu nhiên
    elitism_ratio = 0.2

    best_solution, best_fitness = GA(
        pop_size, generations, crossover_rate, mutation_rate, 
        tournament_size, greedy_ratio, elitism_ratio
    )
    m = 0
    result = []
    for i in range(len(best_solution)):
        if best_solution[i] != -1:
            m += 1
            result.append((i+1, best_solution[i] + 1))
    
    print(m)

    for i, j in result:
        print(f"{i} {j}")

