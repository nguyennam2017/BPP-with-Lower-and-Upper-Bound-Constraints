import random
import math
import copy
import time # Added for time limit
from typing import List, Tuple

class SimulatedAnnealingSearch:
    def __init__(self, N: int, K: int, demands: List[int], costs: List[int], 
                 c1: List[int], c2: List[int],
                 initial_temp: float = 1e5, cooling_rate: float = 0.995, 
                 min_temp: float = 1e-2, max_iterations_no_improvement: int = 5000,
                 time_limit: float = float('inf'), verbose: bool = True): # Added time_limit and verbose
        """
        Initializes the Simulated Annealing search algorithm.

        Args:
            N: Number of orders.
            K: Number of vehicles.
            demands: List of demands for each order (0-indexed).
            costs: List of costs (or profits) for each order (0-indexed).
            c1: List of lower capacity bounds for each vehicle (0-indexed for K vehicles).
            c2: List of upper capacity bounds for each vehicle (0-indexed for K vehicles).
            initial_temp: Starting temperature for SA.
            cooling_rate: Rate at which temperature decreases.
            min_temp: Minimum temperature to stop SA.
            max_iterations_no_improvement: Max iterations without improvement in best solution to stop early.
            time_limit (float): Maximum execution time in seconds.
            verbose (bool): Whether to print progress and solution details.
        """
        self.N = N
        self.K = K
        self.d = demands
        self.c = costs
        self.c1 = [0] + c1
        self.c2 = [0] + c2

        # SA parameters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations_no_improvement = max_iterations_no_improvement
        self.time_limit = time_limit
        self.verbose = verbose

        # Solution representation (initialized in init_solution or solve)
        self.x = [0] * self.N  # Current assignments
        self.load = [0.0] * (self.K + 1) 
        self.cost_contrib = [0.0] * (self.K + 1) # Can be removed if not used elsewhere

        # Objective values
        self.current_obj = 0.0
        self.best_obj = 0.0 # Stores the best objective value found
        self.best_x = [0] * self.N # Stores the assignments of the best solution

        self.iterations_since_last_improvement = 0

    def _calculate_objective(self, assignments: List[int], current_loads: List[float]) -> float:
        """
        Calculates the total objective value (sum of costs from feasible vehicles).
        A vehicle j is feasible if c1[j] <= load[j] <= c2[j].
        """
        total_objective = 0.0
        for j in range(1, self.K + 1): 
            is_feasible = (self.c1[j] <= current_loads[j] <= self.c2[j])
            if is_feasible and current_loads[j] > 0: 
                vehicle_actual_cost = 0
                for i in range(self.N):
                    if assignments[i] == j:
                        vehicle_actual_cost += self.c[i]
                total_objective += vehicle_actual_cost
        return total_objective

    def _init_greedy_solution(self):
        """
        Generates an initial greedy solution.
        Orders are sorted by cost/demand ratio (descending) and greedily assigned.
        Vehicles that are not feasible after initial assignment have their orders unassigned.
        Sets self.x, self.load, self.current_obj, self.best_obj, self.best_x.
        """
        self.load = [0.0] * (self.K + 1)
        self.cost_contrib = [0.0] * (self.K + 1) 
        self.x = [0] * self.N

        order_indices = list(range(self.N))
        order_indices.sort(key=lambda i: self.c[i] / (self.d[i] + 1e-9), reverse=True)

        for i_order_idx in order_indices: 
            assigned = False
            for j_vehicle_idx in range(1, self.K + 1): 
                if self.load[j_vehicle_idx] + self.d[i_order_idx] <= self.c2[j_vehicle_idx]:
                    self.x[i_order_idx] = j_vehicle_idx
                    self.load[j_vehicle_idx] += self.d[i_order_idx]
                    assigned = True
                    break
            if not assigned:
                self.x[i_order_idx] = 0 
        
        temp_load_for_check = [0.0] * (self.K + 1)
        for i in range(self.N):
            vehicle_idx = self.x[i]
            if vehicle_idx != 0:
                temp_load_for_check[vehicle_idx] += self.d[i]

        for j in range(1, self.K + 1):
            if temp_load_for_check[j] > 0 and not (self.c1[j] <= temp_load_for_check[j] <= self.c2[j]):
                for i in range(self.N):
                    if self.x[i] == j:
                        self.x[i] = 0 

        self.load = [0.0] * (self.K + 1)
        # self.cost_contrib = [0.0] * (self.K + 1) # Recalculate if needed, or remove if unused
        for i in range(self.N):
            vehicle_idx = self.x[i]
            if vehicle_idx != 0:
                 self.load[vehicle_idx] += self.d[i]
                 # self.cost_contrib[vehicle_idx] += self.c[i]

        self.current_obj = self._calculate_objective(self.x, self.load)
        self.best_obj = self.current_obj
        self.best_x = self.x[:]
        
        if self.verbose:
            print(f"Initial greedy solution objective: {self.best_obj}")

    def get_solution_details(self, assignments_to_check: List[int]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Calculates and returns the details of a given solution.
        Only orders assigned to feasible vehicles are considered served.

        Args:
            assignments_to_check (List[int]): The list of assignments (0-indexed order to 1-indexed vehicle).

        Returns:
            Tuple: (assignments_output, not_assigned_orders_output)
            assignments_output (List[Tuple[int, int]]): List of (order_id+1, vehicle_id) for served orders.
            not_assigned_orders_output (List[int]): List of 0-indexed order IDs that were not served.
        """
        final_assignments_for_return = [] 
        served_order_indices_0_based = set()
        
        # Calculate loads based on the provided assignments_to_check
        current_solution_loads = [0.0] * (self.K + 1)
        for i in range(self.N):
            vehicle_idx = assignments_to_check[i]
            if vehicle_idx != 0: 
                current_solution_loads[vehicle_idx] += self.d[i]

        for i in range(self.N): 
            vehicle_idx = assignments_to_check[i]
            if vehicle_idx != 0: 
                is_vehicle_feasible = (self.c1[vehicle_idx] <= current_solution_loads[vehicle_idx] <= self.c2[vehicle_idx])
                if is_vehicle_feasible:
                    final_assignments_for_return.append((i + 1, vehicle_idx)) 
                    served_order_indices_0_based.add(i)
        
        not_assigned_orders_0_based = []
        for i in range(self.N):
            if i not in served_order_indices_0_based:
                not_assigned_orders_0_based.append(i)
                
        return final_assignments_for_return, not_assigned_orders_0_based

    def solve(self) -> Tuple[List[Tuple[int, int]], float, List[int]]:
        """
        Orchestrates the Simulated Annealing solving process.
        Initializes a solution, then iteratively improves it.
        Handles time limit and verbosity.

        Returns:
            Tuple: (assignments, best_total_cost, not_assigned)
            assignments (List[Tuple[int, int]]): List of (order_id+1, vehicle_id) for served orders from the best solution.
            best_total_cost (float): The objective value of the best solution found.
            not_assigned (List[int]): List of 0-indexed order IDs that were not served in the best solution.
        """
        start_time = time.time()
        
        self._init_greedy_solution() # Initializes self.x, self.load, self.current_obj, self.best_obj, self.best_x

        # Current state for SA iterations
        current_x_state = self.x[:]
        current_load_state = self.load[:]
        current_obj_state = self.current_obj
        
        # self.best_obj and self.best_x are already set by _init_greedy_solution

        temp = self.initial_temp
        self.iterations_since_last_improvement = 0
        iteration_count = 0

        while temp > self.min_temp:
            if time.time() - start_time > self.time_limit:
                if self.verbose:
                    print(f"Time limit ({self.time_limit}s) exceeded at iteration {iteration_count}. Stopping.")
                break
            
            iteration_count += 1
            
            # Generate a neighbor solution
            order_to_move = random.randint(0, self.N - 1)
            new_vehicle_assignment = random.randint(0, self.K) # 0 for unassign, 1..K for vehicles
            old_vehicle_assignment = current_x_state[order_to_move]

            if new_vehicle_assignment == old_vehicle_assignment:
                temp *= self.cooling_rate
                self.iterations_since_last_improvement +=1
                if self.iterations_since_last_improvement > self.max_iterations_no_improvement and iteration_count > 100: # Allow some initial exploration
                    if self.verbose:
                        print(f"Stopping early at iter {iteration_count} due to no improvement in {self.max_iterations_no_improvement} iters.")
                    break
                continue

            # Create candidate next state (next_x_state, next_load_state)
            next_x_state = current_x_state[:]
            next_load_state = current_load_state[:]
            order_demand = self.d[order_to_move]

            # Update load: remove from old vehicle
            if old_vehicle_assignment != 0:
                next_load_state[old_vehicle_assignment] -= order_demand
            
            # Update assignment and load: add to new vehicle
            next_x_state[order_to_move] = new_vehicle_assignment
            if new_vehicle_assignment != 0:
                next_load_state[new_vehicle_assignment] += order_demand
            
            new_obj_candidate = self._calculate_objective(next_x_state, next_load_state)

            # SA Acceptance criterion
            delta_obj = new_obj_candidate - current_obj_state
            accepted = False
            if delta_obj > 0: # Better solution
                accepted = True
            else: # Worse or equal, accept with probability
                if temp > 1e-9: # Avoid math errors with very small temp
                    acceptance_probability = math.exp(delta_obj / temp)
                    if random.random() < acceptance_probability:
                        accepted = True
            
            if accepted:
                current_x_state = next_x_state
                current_load_state = next_load_state
                current_obj_state = new_obj_candidate

                if current_obj_state > self.best_obj:
                    self.best_obj = current_obj_state
                    self.best_x = current_x_state[:] # Deep copy
                    self.iterations_since_last_improvement = 0
                    if self.verbose and iteration_count % 1000 == 0: # Occasional progress update
                         print(f"Iter {iteration_count}, Temp {temp:.2f}, New best obj: {self.best_obj:.2f}")
                else:
                    self.iterations_since_last_improvement +=1
            else: # Not accepted
                 self.iterations_since_last_improvement +=1

            temp *= self.cooling_rate
            if self.iterations_since_last_improvement > self.max_iterations_no_improvement and iteration_count > 100:
                #if self.verbose:
                    #print(f"Stopping early at iter {iteration_count} due to no improvement in {self.max_iterations_no_improvement} iters.")
                break
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            #print(f"SA completed in {elapsed_time:.2f}s. Final best objective: {self.best_obj}")

        # Get final assignment details from the best solution found
        assignments_output, not_assigned_output = self.get_solution_details(self.best_x)

        if self.verbose:
            served_count_print = len(assignments_output)
            #print(f"Total served orders in best solution: {served_count_print}")
            # Optionally print all assignments if needed, but can be long
            # for order_print_idx, vehicle_print_idx in assignments_output:
            #     print(f"Order {order_print_idx} -> Vehicle {vehicle_print_idx}")

        return assignments_output, self.best_obj, not_assigned_output
