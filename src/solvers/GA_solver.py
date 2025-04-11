# File: solvers/GA_solver.py

import numpy as np
import random
import time
import copy

class GABinPackingSolver:
    def __init__(self, N, K, demands, costs, c1, c2, time_limit=float('inf'), category=None):
        self.N = N
        self.K = K
        self.demands = demands
        self.costs = costs
        self.c1 = c1
        self.c2 = c2
        self.time_limit = time_limit
        self.category = category

        self.pop_size = 50
        self.num_generations = 1000
        self.initial_mutate_rate = 0.2
        self.greedy_ratio = 0.3
        self.max_no_improvement = 50

        # Ma trận theo dõi tần suất gán khả thi cho từng đơn hàng và xe
    def fitness(self, individual):
        quantity = [0] * self.K
        cost = [0] * self.K

        for i in range(self.N):
            trunk = individual[i]
            if trunk != -1:
                quantity[trunk] += self.demands[i]
                cost[trunk] += self.costs[i]

        total_cost = 0
        penaty = 0

        for i in range(self.K):
            if self.c1[i] <= quantity[i] <= self.c2[i]:
                total_cost += cost[i]
            elif quantity[i] < self.c1[i]:
                penaty += 100 * (self.c1[i] - quantity[i])
            else:
                penaty += 100 * (quantity[i] - self.c2[i])

        return total_cost - penaty
    
    def greedy_initializer(self):
        individual = [-1] * self.N
        ratios = [(i,self.costs[i]/self.demands[i]) for i in range(self.N)]

        ratios.sort(key=lambda x: x[1], reverse=True)

        vehicle_load = [0] * self.K
        for order_idx, _ in ratios:
            best_vehicle = -1
            best_remaning = float('inf')

            for v in range(self.K):
                new_load = vehicle_load[v] + self.demands[order_idx]
                # Kiểm tra nếu thêm đơn hàng vẫn nằm trong giới hạn tải
                if self.c1[v] <= new_load <= self.c2[v]:
                    # Tìm xe có dung lượng còn lại ít nhất sau khi gán
                    remaining = self.c2[v] - new_load
                    if remaining < best_remaning:
                        best_remaning = remaining
                        best_vehicle = v

            if best_vehicle != -1:
                individual[order_idx] = best_vehicle
                vehicle_load[best_vehicle] += self.demands[order_idx]


        return individual
    
    def random_initializers(self):
        return [random.randint(-1,self.K-1) for _ in range(self.N)]
    
    def initializers_population(self, pop_size, greedy_ratio = 0.4):
        population = []
        greedy_count = int(pop_size * greedy_ratio)
        for _ in range(greedy_count):
            population.append(self.greedy_initializer())

        for _ in range(pop_size - greedy_count):
            population.append(self.random_initializers())

        return population
    
    def crosserver(self, p1, p2):
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
    
    def mutate(self, individual, mutation_rate=0.1):
        mutated = individual.copy()

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutation_type = random.choice(['reassign', 'swap', 'remove', 'neighbor'])

                if mutation_type == 'reassign':
                    mutated[i] = random.randint(-1, self.K - 1)

                elif mutation_type == 'remove':
                    mutated[i] = -1

                elif mutation_type == 'neighbor':
                    if mutated[i] != -1:
                        shift = random.choice([-1, 1])
                        new_val = mutated[i] + shift
                        if 0 <= new_val < self.K:
                            mutated[i] = new_val

                elif mutation_type == 'swap':
                    j = random.randint(0, len(mutated) - 1)
                    mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated
    
    def tournament_selection(self, population, fitness_values, tournament_size=3):
        tournament_indices = random.sample(range(len(population)), tournament_size)

        best_idx = tournament_indices[0]

        for i in tournament_indices:
            if fitness_values[i] > fitness_values[best_idx]:
                best_idx = i

        return population[best_idx]
    
    def GA(self, pop_size = 100, generations = 1000, crosserver_rate=0.7, mutaion_rate=0.2, tourrnaments_size=3, greedy_ratio=0.4, elitism=0.2):
        population = self.initializers_population(pop_size=pop_size, greedy_ratio=greedy_ratio)

        best_individual = None
        best_fitness = float('-inf')
        
        # Theo dõi số thế hệ liên tiếp không cải thiện
        no_improvement_count = 0
        
        start_time = time.time()
        
        for gen in range(generations):
            # Kiểm tra giới hạn thời gian
            if time.time() - start_time > self.time_limit:
                break
                
            fitness_values = [self.fitness(ind) for ind in population]

            sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_values[i] for i in sorted_indices]

            if sorted_fitness[0] > best_fitness:
                best_fitness = sorted_fitness[0]
                best_individual = sorted_population[0].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Dừng nếu không cải thiện sau nhiều thế hệ
            if no_improvement_count >= self.max_no_improvement:
                break

            new_populatiom = []

            elitism_count = int(pop_size * elitism)
            new_populatiom.extend([ind.copy() for ind in sorted_population[:elitism_count]])


            while len(new_populatiom) < pop_size:
                p1 = self.tournament_selection(population, fitness_values, tourrnaments_size)
                p2 = self.tournament_selection(population, fitness_values, tourrnaments_size)

                if random.random() < crosserver_rate:
                    child1, child2 = self.crosserver(p1, p2)
                else:
                    child1, child2 = p1.copy(), p2.copy()
                
                # Áp dụng đột biến
                child1 = self.mutate(child1, mutaion_rate)
                child2 = self.mutate(child2, mutaion_rate)

                new_populatiom.append(child1)
                if len(new_populatiom) < pop_size:
                    new_populatiom.append(child2)

            population = new_populatiom

        final_fitness_value = [self.fitness(ind) for ind in population]
        final_best_idx = final_fitness_value.index(max(final_fitness_value))

        if final_fitness_value[final_best_idx] > best_fitness:
            best_individual = population[final_best_idx].copy()
            best_fitness = final_fitness_value[final_best_idx]

        return best_individual, best_fitness
    
    def extract_solution(self, individual):
        """
        Chuyển đổi một cá thể (individual) sang định dạng lời giải [(order_id, vehicle_id), ...]
        """
        assignments = []
        for i in range(self.N):
            if individual[i] != -1:  # Chỉ thêm những đơn hàng được gán
                assignments.append((i, individual[i]))
        return assignments
    
    def calculate_total_cost(self, assignments):
        """
        Tính tổng chi phí cho một giải pháp
        """
        # Khởi tạo mảng theo dõi
        quantity = [0] * self.K
        cost = [0] * self.K
        
        # Cập nhật lượng hàng và chi phí cho mỗi xe
        for order_id, vehicle_id in assignments:
            quantity[vehicle_id] += self.demands[order_id]
            cost[vehicle_id] += self.costs[order_id]
        
        # Tính tổng chi phí cho các xe có lượng hàng nằm trong khoảng hợp lệ
        total_cost = 0
        for i in range(self.K):
            if self.c1[i] <= quantity[i] <= self.c2[i]:
                total_cost += cost[i]
        
        return total_cost
    
    def solve(self):
        """
        Giải bài toán và trả về kết quả theo định dạng yêu cầu
        
        Returns:
            tuple: (assignments, total_cost, extra_info) trong đó:
                - assignments: danh sách các cặp (order_id, vehicle_id)
                - total_cost: tổng chi phí
                - extra_info: thông tin bổ sung (dict)
        """
        start_time = time.time()
        
        # Chạy thuật toán di truyền
        best_individual, best_fitness = self.GA(
            pop_size=self.pop_size,
            generations=self.num_generations,
            mutaion_rate=self.initial_mutate_rate,
            greedy_ratio=self.greedy_ratio
        )
        
        # Chuyển đổi cá thể tốt nhất sang định dạng lời giải
        assignments = self.extract_solution(best_individual)
        
        # Tính tổng chi phí
        total_cost = self.calculate_total_cost(assignments)
        
        # Thông tin bổ sung
        extra_info = {
            "runtime": time.time() - start_time,
            "fitness": best_fitness
        }
        
        return assignments, total_cost, extra_info