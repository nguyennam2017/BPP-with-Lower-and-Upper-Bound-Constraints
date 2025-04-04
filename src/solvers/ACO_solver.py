import numpy as np
import random

class ACOBinPackingSolver:
    def __init__(self, N, K, demands, costs, c1, c2,
                 num_ants=100, num_iterations=100,
                 alpha=1, beta=2, evaporation=0.5, Q=100, 
                 elite_ants=5, verbose=True):
        self.N = N
        self.K = K
        self.demands = demands
        self.costs = costs
        self.c1 = c1
        self.c2 = c2
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.elite_ants = min(elite_ants, num_ants)  # Số kiến tinh hoa (không vượt quá num_ants)
        self.verbose = verbose

        # Khởi tạo ma trận pheromone
        self.pheromones = np.ones((self.N, self.K))

    def _calculate_heuristic(self):
        """Tính heuristic dựa trên chi phí."""
        heuristic = np.zeros((self.N, self.K))
        for i in range(self.N):
            for k in range(self.K):
                if self.c1[k] <= self.demands[i] <= self.c2[k]:
                    heuristic[i][k] = self.costs[i]  # Ưu tiên đơn hàng có chi phí cao
        return heuristic

    def _select_vehicle(self, pheromone_row, heuristic_row):
        """Chọn xe dựa trên xác suất pheromone và heuristic."""
        probabilities = (pheromone_row ** self.alpha) * (heuristic_row ** self.beta)
        total = np.sum(probabilities)

        if total == 0 or np.isnan(total):
            valid_indices = [k for k in range(self.K) if heuristic_row[k] > 0]
            if valid_indices:
                return random.choice(valid_indices)
            return -1  # Không gán xe
        probabilities /= total
        return np.random.choice(range(self.K), p=probabilities)

    def solve(self):
        """Giải bài toán bằng ACO với kiến tinh hoa để tối đa hóa tổng chi phí."""
        best_solution = None
        best_cost = 0

        heuristic = self._calculate_heuristic()

        for iteration in range(self.num_iterations):
            solutions = []  # Danh sách (assignment, cost) của từng kiến

            for ant in range(self.num_ants):
                assignment = [-1] * self.N  # -1 nghĩa là chưa gán
                vehicle_loads = [0] * self.K
                total_cost = 0

                # Xáo trộn thứ tự đơn hàng để tăng tính khám phá
                order_indices = list(range(self.N))
                random.shuffle(order_indices)

                for i in order_indices:
                    k = self._select_vehicle(self.pheromones[i], heuristic[i])
                    if k != -1 and self.c1[k] <= vehicle_loads[k] + self.demands[i] <= self.c2[k]:
                        assignment[i] = k
                        vehicle_loads[k] += self.demands[i]
                        total_cost += self.costs[i]

                solutions.append((assignment, total_cost))
                if total_cost > best_cost:
                    best_cost = total_cost
                    best_solution = assignment.copy()

            # Sắp xếp các giải pháp theo chi phí giảm dần
            solutions.sort(key=lambda x: x[1], reverse=True)

            # Cập nhật pheromone: bay hơi trước
            self.pheromones *= self.evaporation

            # Chỉ "kiến tinh hoa" (elite_ants) được cập nhật pheromone
            for assignment, cost in solutions[:self.elite_ants]:
                for i, k in enumerate(assignment):
                    if k != -1:
                        self.pheromones[i][k] += self.Q * cost  # Cộng pheromone theo chi phí

        # Xử lý trường hợp không tìm thấy giải pháp
        if best_solution is None:
            return [], 0, list(range(1, self.N + 1))

        assignments = [(i + 1, k + 1) for i, k in enumerate(best_solution) if k != -1]
        not_assigned = [i + 1 for i, k in enumerate(best_solution) if k == -1]
        return assignments, best_cost, not_assigned

