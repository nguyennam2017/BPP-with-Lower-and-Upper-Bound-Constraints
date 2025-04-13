import numpy as np
import random
import time
import copy
from collections import Counter

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

        # GA parameters
        self.pop_size = 50
        self.num_generations = 200000
        self.initial_mutation_rate = 0.2
        self.final_mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.greedy_ratio = 0.3
        self.max_no_improvement = 50
        self.tournament_size = 3
        self.init_elitism = 0.2
        self.selection_methods = ["tournament", "roulette", "rank"]
        self.crossover_methods = ["single_point", "two_point", "uniform"]
        self.restart_threshold = 20

    def fitness(self, individual):
        quantity = [0] * self.K
        cost = [0] * self.K

        for i in range(self.N):
            trunk = individual[i]
            if trunk != -1:
                quantity[trunk] += self.demands[i]
                cost[trunk] += self.costs[i]

        total_cost = 0
        penalty = 0

        for i in range(self.K):
            if quantity[i] > 0:  # Only calculate for used vehicles
                if self.c1[i] <= quantity[i] <= self.c2[i]:
                    total_cost += cost[i]
                elif quantity[i] < self.c1[i]:
                    # Non-linear penalty for underloaded vehicles
                    deficit = self.c1[i] - quantity[i]
                    penalty += 100 * deficit * (1 + deficit / max(1, self.c1[i]))
                else:
                    # Non-linear penalty for overloaded vehicles
                    excess = quantity[i] - self.c2[i]
                    penalty += 100 * excess * (1 + excess / max(1, self.c2[i]))

        # Additional penalty for unused orders
        unassigned = individual.count(-1)
        if unassigned > 0:
            penalty += 50 * unassigned

        return total_cost - penalty

    def greedy_initializer(self, randomize=False):
        """Greedy initialization with optional randomization"""
        individual = [-1] * self.N
        
        # Handle special case: prevent zero division - orders with zero demand
        safe_demands = [max(1, d) for d in self.demands]
        
        # Calculate value/weight ratios for all orders
        ratios = [(i, self.costs[i] / safe_demands[i]) for i in range(self.N)]
        
        if randomize:
            # Add some noise to the ratios to create diversity
            ratios = [(i, ratio * random.uniform(0.9, 1.1)) for i, ratio in ratios]
        
        # Sort by ratio (highest value/weight first)
        ratios = sorted(ratios, key=lambda x: x[1], reverse=True)

        vehicle_load = [0] * self.K
        for order_idx, _ in ratios:
            best_vehicle = -1
            best_score = float('-inf')

            # Try to assign to each vehicle
            for v in range(self.K):
                new_load = vehicle_load[v] + self.demands[order_idx]
                
                # Calculate a score based on how well this fits
                if new_load <= self.c2[v]:
                    # Prefer vehicles that are already in use
                    usage_bonus = 10 if vehicle_load[v] > 0 else 0
                    
                    # Handle special case: c1[v] == c2[v]
                    if self.c1[v] == self.c2[v]:
                        # In this case, only assign if it's an exact match
                        if new_load == self.c1[v]:
                            score = 5 + usage_bonus  # High score for exact match
                        else:
                            score = -1  # Low score otherwise
                    else:
                        # Score is better when closer to lower capacity bound
                        if new_load < self.c1[v]:
                            score = (new_load / max(1, self.c1[v])) + usage_bonus
                        else:
                            # Perfect fit between c1 and c2
                            score = 2 + usage_bonus - (new_load - self.c1[v]) / max(1, (self.c2[v] - self.c1[v]))
                    
                    if score > best_score:
                        best_score = score
                        best_vehicle = v

            if best_vehicle != -1:
                individual[order_idx] = best_vehicle
                vehicle_load[best_vehicle] += self.demands[order_idx]

        return individual

    def first_fit_initializer(self):
        """First-fit descending initialization"""
        individual = [-1] * self.N
        
        # Sort orders by demand (descending)
        sorted_orders = sorted(range(self.N), key=lambda i: self.demands[i], reverse=True)
        
        vehicle_load = [0] * self.K
        for order_idx in sorted_orders:
            # Find the first vehicle that can fit this order
            for v in range(self.K):
                new_load = vehicle_load[v] + self.demands[order_idx]
                if new_load <= self.c2[v]:
                    individual[order_idx] = v
                    vehicle_load[v] = new_load
                    break
                    
        return individual

    def best_fit_initializer(self):
        """Best-fit descending initialization"""
        individual = [-1] * self.N
        
        # Sort orders by demand (descending)
        sorted_orders = sorted(range(self.N), key=lambda i: self.demands[i], reverse=True)
        
        vehicle_load = [0] * self.K
        for order_idx in sorted_orders:
            best_vehicle = -1
            min_remaining = float('inf')
            
            # Find the vehicle with minimum remaining capacity after assignment
            for v in range(self.K):
                new_load = vehicle_load[v] + self.demands[order_idx]
                if new_load <= self.c2[v]:
                    # Special case: if c1[v] == c2[v], prioritize exact matches
                    if self.c1[v] == self.c2[v]:
                        if new_load == self.c1[v]:
                            best_vehicle = v
                            break
                    # Normal case
                    elif self.c1[v] <= new_load:
                        remaining = self.c2[v] - new_load
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_vehicle = v
                    # If no vehicles that meet c1 constraint, consider under-loaded ones
                    elif best_vehicle == -1:
                        remaining = self.c2[v] - new_load
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_vehicle = v
            
            if best_vehicle != -1:
                individual[order_idx] = best_vehicle
                vehicle_load[best_vehicle] += self.demands[order_idx]
                
        return individual

    def random_initializer(self):
        """Random initialization with bias toward legal assignments"""
        individual = [-1] * self.N
        vehicle_load = [0] * self.K
        
        # Process orders in random order
        order_indices = list(range(self.N))
        random.shuffle(order_indices)
        
        for i in order_indices:
            # 15% chance of not assigning the order
            if random.random() < 0.15:
                continue
                
            # Try to find a legal assignment
            valid_vehicles = [v for v in range(self.K) if vehicle_load[v] + self.demands[i] <= self.c2[v]]
            
            if valid_vehicles:
                # Choose a random valid vehicle
                v = random.choice(valid_vehicles)
                individual[i] = v
                vehicle_load[v] += self.demands[i]
                
        return individual

    def initialize_population(self, pop_size, greedy_ratio=0.4):
        """Initialize population with multiple initialization strategies"""
        population = []
        
        # Calculate how many individuals to create with each method
        greedy_count = int(pop_size * greedy_ratio * 0.5)
        greedy_random_count = int(pop_size * greedy_ratio * 0.5)
        first_fit_count = int(pop_size * (1 - greedy_ratio) * 0.3)
        best_fit_count = int(pop_size * (1 - greedy_ratio) * 0.3)
        random_count = pop_size - greedy_count - greedy_random_count - first_fit_count - best_fit_count
        
        # Create individuals with different methods
        for _ in range(greedy_count):
            population.append(self.greedy_initializer(randomize=False))
            
        for _ in range(greedy_random_count):
            population.append(self.greedy_initializer(randomize=True))
            
        for _ in range(first_fit_count):
            population.append(self.first_fit_initializer())
            
        for _ in range(best_fit_count):
            population.append(self.best_fit_initializer())
            
        for _ in range(random_count):
            population.append(self.random_initializer())
            
        return population

    def single_point_crossover(self, p1, p2):
        """Single-point crossover"""
        if len(p1) < 2:
            return p1.copy(), p2.copy()

        point = random.randint(1, len(p1) - 1)
        child1 = p1[:point] + p2[point:]
        child2 = p2[:point] + p1[point:]
        
        return child1, child2

    def two_point_crossover(self, p1, p2):
        """Two-point crossover"""
        if len(p1) < 3:
            return p1.copy(), p2.copy()

        point1 = random.randint(1, len(p1) - 2)
        point2 = random.randint(point1 + 1, len(p1) - 1)

        child1 = p1[:point1] + p2[point1:point2] + p1[point2:]
        child2 = p2[:point1] + p1[point1:point2] + p2[point2:]
        
        return child1, child2

    def uniform_crossover(self, p1, p2):
        """Uniform crossover"""
        child1, child2 = [], []
        
        for i in range(len(p1)):
            if random.random() < 0.5:
                child1.append(p1[i])
                child2.append(p2[i])
            else:
                child1.append(p2[i])
                child2.append(p1[i])
                
        return child1, child2

    def crossover(self, p1, p2, method=None):
        """Perform crossover using the specified method"""
        if method is None:
            method = random.choice(self.crossover_methods)
            
        if method == "single_point":
            return self.single_point_crossover(p1, p2)
        elif method == "two_point":
            return self.two_point_crossover(p1, p2)
        elif method == "uniform":
            return self.uniform_crossover(p1, p2)
        else:
            return self.two_point_crossover(p1, p2)

    def mutate(self, individual, mutation_rate=0.1):
        """Improved mutation with multiple strategies"""
        mutated = individual.copy()
        
        # Apply different mutation types with varying probabilities
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Choose mutation type with different probabilities
                r = random.random()
                
                if r < 0.3:  # Reassign to a different vehicle
                    # Prefer vehicles that are already in use
                    used_vehicles = [v for v in range(self.K) if v in mutated and v != -1]
                    
                    if used_vehicles and random.random() < 0.7:
                        mutated[i] = random.choice(used_vehicles) if used_vehicles else random.randint(-1, self.K - 1)
                    else:
                        mutated[i] = random.randint(-1, self.K - 1)
                        
                elif r < 0.5:  # Swap with another order
                    j = random.randint(0, len(mutated) - 1)
                    mutated[i], mutated[j] = mutated[j], mutated[i]
                    
                elif r < 0.7:  # Remove assignment
                    mutated[i] = -1
                    
                elif r < 0.9:  # Move to neighbor vehicle
                    if mutated[i] != -1:
                        neighbors = []
                        
                        # Find valid neighbor vehicles
                        if mutated[i] > 0:
                            neighbors.append(mutated[i] - 1)
                        if mutated[i] < self.K - 1:
                            neighbors.append(mutated[i] + 1)
                            
                        if neighbors:
                            mutated[i] = random.choice(neighbors)
                else:
                    # Balance mutation - try to balance vehicle loads
                    # Calculate current vehicle loads
                    loads = [0] * self.K
                    for idx, v in enumerate(mutated):
                        if v != -1 and idx != i:  # Skip the current order
                            loads[v] += self.demands[idx]
                    
                    # Find the least loaded vehicle that can accept this order
                    valid_vehicles = []
                    for v in range(self.K):
                        if loads[v] + self.demands[i] <= self.c2[v]:
                            valid_vehicles.append((v, loads[v]))
                    
                    if valid_vehicles:
                        # Sort by load
                        valid_vehicles.sort(key=lambda x: x[1])
                        # Randomly choose one of the least loaded vehicles
                        top_n = min(3, len(valid_vehicles))
                        mutated[i] = valid_vehicles[random.randint(0, top_n-1)][0]
        
        # Local search - try to improve the solution
        if random.random() < 0.1:  # 10% chance to apply local search
            mutated = self.local_search(mutated)
            
        return mutated

    def local_search(self, individual):
        """Simple local search to improve solution"""
        improved = individual.copy()
        
        # Calculate current vehicle loads
        vehicle_loads = [0] * self.K
        for i, v in enumerate(improved):
            if v != -1:
                vehicle_loads[v] += self.demands[i]
        
        # Try to move orders from overloaded vehicles to underloaded ones
        for i in range(self.N):
            v = improved[i]
            if v != -1 and vehicle_loads[v] > self.c2[v]:
                # This order is in an overloaded vehicle
                # Try to move it to a vehicle with capacity
                for new_v in range(self.K):
                    if new_v != v and vehicle_loads[new_v] + self.demands[i] <= self.c2[v]:
                        # Move the order
                        vehicle_loads[v] -= self.demands[i]
                        vehicle_loads[new_v] += self.demands[i]
                        improved[i] = new_v
                        break
        
        # Try to assign unassigned orders
        for i in range(self.N):
            if improved[i] == -1:
                # Find a vehicle with capacity
                for v in range(self.K):
                    if vehicle_loads[v] + self.demands[i] <= self.c2[v]:
                        improved[i] = v
                        vehicle_loads[v] += self.demands[i]
                        break
        
        return improved

    def tournament_selection(self, population, fitness_values, tournament_size=3):
        """Tournament selection"""
        # Ensure tournament size is valid
        actual_tournament_size = min(tournament_size, len(population))
        if actual_tournament_size <= 0:
            return random.choice(population) if population else None
            
        indices = random.sample(range(len(population)), actual_tournament_size)
        best = max(indices, key=lambda idx: fitness_values[idx])
        return population[best]

    def roulette_wheel_selection(self, population, fitness_values):
        """Roulette wheel (fitness proportionate) selection"""
        if not population:
            return None
            
        # Adjust fitness values to ensure all are positive
        min_fitness = min(fitness_values)
        adjusted_fitness = [f - min_fitness + 1 for f in fitness_values]
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choice(population)
        
        r = random.uniform(0, total_fitness)
        cumulative = 0
        for i, fitness in enumerate(adjusted_fitness):
            cumulative += fitness
            if cumulative >= r:
                return population[i]
        
        return population[-1]

    def rank_selection(self, population, fitness_values):
        """Rank-based selection"""
        if not population:
            return None
            
        # Create ranks (higher fitness = higher rank)
        ranks = list(range(1, len(population) + 1))
        
        # Sort indices by fitness
        sorted_indices = sorted(range(len(fitness_values)), 
                               key=lambda i: fitness_values[i])
        
        # Assign ranks
        rank_mapping = {idx: rank for idx, rank in zip(sorted_indices, ranks)}
        
        # Calculate total rank sum
        rank_sum = sum(ranks)
        
        # Select based on rank proportion
        r = random.uniform(0, rank_sum)
        cumulative = 0
        for i in range(len(population)):
            cumulative += rank_mapping[i]
            if cumulative >= r:
                return population[i]
        
        return population[-1]

    def select_parent(self, population, fitness_values, method=None):
        """Select a parent using the specified method"""
        if not population:
            return None
            
        if method is None:
            method = random.choice(self.selection_methods)
            
        if method == "tournament":
            return self.tournament_selection(population, fitness_values, self.tournament_size)
        elif method == "roulette":
            return self.roulette_wheel_selection(population, fitness_values)
        elif method == "rank":
            return self.rank_selection(population, fitness_values)
        else:
            return self.tournament_selection(population, fitness_values, self.tournament_size)

    def GA(self, pop_size=100, generations=1000, crossover_rate=0.7,
           initial_mutation_rate=0.2, final_mutation_rate=0.05,
           tournament_size=3, greedy_ratio=0.4, initial_elitism=0.2):
        """Improved Genetic Algorithm implementation"""
        # Handle edge cases
        if pop_size <= 0:
            pop_size = 50
        if generations <= 0:
            generations = 1000
        if crossover_rate < 0 or crossover_rate > 1:
            crossover_rate = 0.7
        if initial_mutation_rate < 0 or initial_mutation_rate > 1:
            initial_mutation_rate = 0.2
        if final_mutation_rate < 0 or final_mutation_rate > 1:
            final_mutation_rate = 0.05
        if tournament_size <= 0:
            tournament_size = 3
        if greedy_ratio < 0 or greedy_ratio > 1:
            greedy_ratio = 0.4
        if initial_elitism < 0 or initial_elitism > 1:
            initial_elitism = 0.2
            
        # Initialize parameters
        self.tournament_size = tournament_size
        
        try:
            population = self.initialize_population(pop_size, greedy_ratio)
        except Exception as e:
            # Fallback to simpler initialization
            print(f"Warning: Error in population initialization: {e}. Falling back to simple initialization.")
            population = [self.random_initializer() for _ in range(pop_size)]
            
        best_individual = None
        best_fitness = float('-inf')
        no_improvement = 0
        restart_counter = 0
        start_time = time.time()
        
        # Dynamic parameters
        elitism_rate = initial_elitism
        
        for gen in range(generations):
            # Check time limit
            if time.time() - start_time > self.time_limit:
                break
                
            # Calculate fitness for current population
            try:
                fitness_values = [self.fitness(ind) for ind in population]
            except Exception as e:
                print(f"Warning: Error in fitness calculation: {e}")
                break
                
            # Check for valid fitness values
            if not fitness_values or all(f == float('-inf') for f in fitness_values):
                print("Warning: No valid solutions found in population.")
                break
                
            # Sort population by fitness
            sorted_indices = sorted(range(len(fitness_values)), 
                                  key=lambda i: fitness_values[i], 
                                  reverse=True)
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_values[i] for i in sorted_indices]
            
            # Update best individual
            if sorted_fitness[0] > best_fitness:
                best_fitness = sorted_fitness[0]
                best_individual = sorted_population[0].copy()
                no_improvement = 0
                restart_counter = 0
            else:
                no_improvement += 1
                restart_counter += 1
            
            # Early stopping
            if no_improvement >= self.max_no_improvement:
                break
                
            # Restart if stuck in local optimum
            if restart_counter >= self.restart_threshold:
                # Keep the best individual(s)
                elite_count = max(1, int(pop_size * 0.1))
                new_population = [ind.copy() for ind in sorted_population[:elite_count]]
                
                # Create new individuals
                while len(new_population) < pop_size:
                    if random.random() < 0.5:
                        try:
                            new_population.append(self.greedy_initializer(randomize=True))
                        except Exception:
                            new_population.append(self.random_initializer())
                    else:
                        new_population.append(self.random_initializer())
                
                population = new_population
                restart_counter = 0
                continue
            
            # Calculate adaptive parameters
            progress = min(1.0, gen / max(1, (generations * 0.7)))  # Progress ratio (capped at 1.0)
            
            # Adaptive mutation rate - decrease from initial to final rate
            mutation_rate = initial_mutation_rate - progress * (initial_mutation_rate - final_mutation_rate)
            
            # Adaptive elitism - increase slightly as evolution progresses
            elitism_rate = initial_elitism + progress * 0.1
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(1, int(pop_size * elitism_rate))
            new_population.extend([ind.copy() for ind in sorted_population[:elite_count]])
            
            # Fill the rest of the population
            while len(new_population) < pop_size:
                # Select parents
                # Use different selection methods with varying probabilities based on progress
                selection_method = None
                if progress < 0.3:
                    # Early stages - more exploration
                    selection_probs = {"tournament": 0.5, "roulette": 0.3, "rank": 0.2}
                elif progress < 0.7:
                    # Middle stages - balanced
                    selection_probs = {"tournament": 0.6, "roulette": 0.2, "rank": 0.2}
                else:
                    # Late stages - more exploitation
                    selection_probs = {"tournament": 0.7, "roulette": 0.1, "rank": 0.2}
                
                r = random.random()
                cumulative = 0
                for method, prob in selection_probs.items():
                    cumulative += prob
                    if r <= cumulative:
                        selection_method = method
                        break
                
                try:
                    p1 = self.select_parent(population, fitness_values, method=selection_method)
                    p2 = self.select_parent(population, fitness_values, method=selection_method)
                    
                    if p1 is None or p2 is None:
                        raise ValueError("Parent selection failed")
                        
                    # Crossover
                    if random.random() < crossover_rate:
                        # Choose crossover method based on progress
                        if progress < 0.3:
                            # Early stages - more exploration with uniform crossover
                            crossover_probs = {"single_point": 0.3, "two_point": 0.3, "uniform": 0.4}
                        elif progress < 0.7:
                            # Middle stages - balanced
                            crossover_probs = {"single_point": 0.3, "two_point": 0.4, "uniform": 0.3}
                        else:
                            # Late stages - more exploitation with more targeted crossover
                            crossover_probs = {"single_point": 0.4, "two_point": 0.4, "uniform": 0.2}
                        
                        r = random.random()
                        cumulative = 0
                        crossover_method = None
                        for method, prob in crossover_probs.items():
                            cumulative += prob
                            if r <= cumulative:
                                crossover_method = method
                                break
                        
                        child1, child2 = self.crossover(p1, p2, method=crossover_method)
                    else:
                        child1, child2 = p1.copy(), p2.copy()
                    
                    # Mutation and add to new population
                    new_population.append(self.mutate(child1, mutation_rate))
                    if len(new_population) < pop_size:
                        new_population.append(self.mutate(child2, mutation_rate))
                except Exception as e:
                    print(f"Warning: Error in selection/crossover/mutation: {e}")
                    # Add random individuals as fallback
                    new_population.append(self.random_initializer())
                    if len(new_population) < pop_size:
                        new_population.append(self.random_initializer())
            
            # Replace old population
            population = new_population
        
        # Final evaluation
        try:
            fitness_values = [self.fitness(ind) for ind in population]
            best_idx = fitness_values.index(max(fitness_values))
            if fitness_values[best_idx] > best_fitness:
                best_individual = population[best_idx].copy()
                best_fitness = fitness_values[best_idx]
        except Exception as e:
            print(f"Warning: Error in final evaluation: {e}")
            # If we haven't found a solution yet, create a random one
            if best_individual is None:
                best_individual = self.random_initializer()
                best_fitness = self.fitness(best_individual)
        
        # Ensure we return a valid solution even if something went wrong
        if best_individual is None:
            best_individual = self.random_initializer()
            best_fitness = self.fitness(best_individual)
            
        return best_individual, best_fitness

    def extract_solution(self, individual):
        """Convert individual to solution format"""
        return [(i, individual[i]) for i in range(self.N) if individual[i] != -1]

    def calculate_total_cost(self, assignments):
        """Calculate total cost of the solution"""
        quantity = [0] * self.K
        cost = [0] * self.K
        for order_id, vehicle_id in assignments:
            quantity[vehicle_id] += self.demands[order_id]
            cost[vehicle_id] += self.costs[order_id]

        total_cost = 0
        for i in range(self.K):
            if self.c1[i] <= quantity[i] <= self.c2[i]:
                total_cost += cost[i]
        return total_cost

    def solve(self):
        """Main solving method - keep this interface unchanged"""
        start_time = time.time()
        
        try:
            best_individual, best_fitness = self.GA(
                pop_size=self.pop_size,
                generations=self.num_generations,
                crossover_rate=self.crossover_rate,
                initial_mutation_rate=self.initial_mutation_rate,
                final_mutation_rate=self.final_mutation_rate,
                tournament_size=self.tournament_size,
                greedy_ratio=self.greedy_ratio,
                initial_elitism=self.init_elitism
            )
            
            assignments = self.extract_solution(best_individual)
            total_cost = self.calculate_total_cost(assignments)
            
            # Additional statistics for extra_info
            assigned_orders = len(assignments)
            total_orders = self.N
            assignment_ratio = assigned_orders / total_orders if total_orders > 0 else 0
            
            vehicle_loads = [0] * self.K
            for order_id, vehicle_id in assignments:
                vehicle_loads[vehicle_id] += self.demands[order_id]
                
            used_vehicles = sum(1 for load in vehicle_loads if load > 0)
            load_balance = 0 if used_vehicles <= 1 else np.std([load for load in vehicle_loads if load > 0])
            
        except Exception as e:
            print(f"Error in solve method: {e}")
            # Return a minimal valid solution in case of error
            assignments = []
            total_cost = 0
            used_vehicles = 0
            load_balance = 0
            assignment_ratio = 0
            best_fitness = float('-inf')
        
        extra_info = {
            "runtime": time.time() - start_time,
            "fitness": best_fitness,
            "assigned_orders": len(assignments),
            "assignment_ratio": assignment_ratio,
            "used_vehicles": used_vehicles,
            "load_balance": load_balance
        }
    
        return assignments, total_cost, extra_info