import numpy as np
import random
import time

class ACOBinPackingSolver:
    def __init__(self, N, K, demands, costs, c1, c2,
                 num_ants=1000, num_iterations=100,
                 alpha=1.0, beta=1.0, evaporation=0.5, Q=100.0,
                 elite_ants_rate=0.1, # Tỷ lệ kiến tinh hoa, ví dụ 10%
                 time_limit=float('inf'), verbose=True,
                 tau_min_factor=0.01, tau_max_factor=20.0): # Thêm cho MMAS (tùy chọn)
        """
        Khởi tạo ACO solver.

        Args:
            N (int): Số lượng đơn hàng.
            K (int): Số lượng phương tiện.
            demands (list): Danh sách khối lượng của các đơn hàng.
            costs (list): Danh sách giá trị (chi phí) của các đơn hàng.
            c1 (list): Danh sách dung lượng cận dưới của các phương tiện.
            c2 (list): Danh sách dung lượng cận trên của các phương tiện.
            num_ants (int): Số lượng kiến.
            num_iterations (int): Số lượng vòng lặp.
            alpha (float): Trọng số của pheromone.
            beta (float): Trọng số của heuristic.
            evaporation (float): Tốc độ bay hơi pheromone.
            Q (float): Hằng số cập nhật pheromone.
            elite_ants_rate (float): Tỷ lệ kiến tinh hoa so với tổng số kiến.
            time_limit (float): Giới hạn thời gian chạy (giây).
            verbose (bool): True để in thông tin chi tiết.
            tau_min_factor (float): Hệ số cho tau_min (sử dụng nếu áp dụng MMAS).
            tau_max_factor (float): Hệ số cho tau_max (sử dụng nếu áp dụng MMAS).
        """
        self.N = N
        self.K = K
        self.demands = np.array(demands)
        self.costs = np.array(costs)
        self.c1 = np.array(c1)
        self.c2 = np.array(c2)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.elite_ants = max(1, int(num_ants * elite_ants_rate)) # Đảm bảo ít nhất 1 kiến tinh hoa
        self.time_limit = time_limit
        self.verbose = verbose

        # Khởi tạo ma trận pheromone
        # Giá trị khởi tạo ban đầu có thể ảnh hưởng đến khám phá ban đầu.
        # Một cách phổ biến là tau0 = 1.0 / (N * (average_cost or heuristic_value))
        # Ở đây dùng giá trị đơn giản là 1.0
        self.pheromones = np.ones((self.N, self.K)) * 1.0
        self.heuristic = self._calculate_heuristic()

        # Cho MMAS (Max-Min Ant System) - Tùy chọn, có thể kích hoạt sau
        self.apply_mmas = False # Đặt là True để thử nghiệm MMAS
        self.tau_min = None
        self.tau_max = None
        self.tau_min_factor = tau_min_factor
        self.tau_max_factor = tau_max_factor


    def _calculate_heuristic(self):
        """Tính heuristic dựa trên chi phí và khả năng chứa của xe."""
        heuristic = np.zeros((self.N, self.K)) # Khởi tạo bằng 0
        for i in range(self.N):
            for k in range(self.K):
                if self.demands[i] <= self.c2[k]: 
                    heuristic[i][k] = self.costs[i] + 1e-9
        return heuristic

    def _select_vehicle(self, item_idx, pheromone_row, heuristic_row, current_vehicle_loads):
        """
        Chọn xe cho một đơn hàng dựa trên pheromone, heuristic và các ràng buộc.
        Chỉ xem xét các xe mà việc thêm đơn hàng này không vi phạm c2.
        """
        probabilities = np.zeros(self.K)
        valid_vehicles_exist = False

        for k in range(self.K):
            # Chỉ xem xét xe k nếu việc thêm đơn hàng item_idx không làm vượt quá c2[k]
            if current_vehicle_loads[k] + self.demands[item_idx] <= self.c2[k] and heuristic_row[k] > 0:
                # Kiểm tra xem heuristic_row[k] > 0 để đảm bảo xe này có thể chứa đơn hàng (theo _calculate_heuristic)
                probabilities[k] = (pheromone_row[k] ** self.alpha) * \
                                   (heuristic_row[k] ** self.beta)
                valid_vehicles_exist = True
            # else: probabilities[k] vẫn là 0

        if not valid_vehicles_exist or np.sum(probabilities) == 0:
            # Nếu không có xe nào hợp lệ hoặc tổng xác suất bằng 0
            # Thử chọn ngẫu nhiên một xe có thể chứa (không vi phạm c2 và heuristic > 0)
            possible_vehicles = [k_idx for k_idx in range(self.K)
                                 if current_vehicle_loads[k_idx] + self.demands[item_idx] <= self.c2[k_idx] and heuristic_row[k_idx] > 0]
            if possible_vehicles:
                return random.choice(possible_vehicles)
            return -1 # Không gán được xe

        probabilities /= np.sum(probabilities)
        return np.random.choice(range(self.K), p=probabilities)

    def _validate_and_calculate_cost(self, assignment_candidate):
        """
        Xác thực giải pháp dựa trên ràng buộc c1 và c2, và tính chi phí thực tế.
        Một phương tiện chỉ được coi là hợp lệ nếu tổng tải trọng của nó
        nằm trong [c1[k], c2[k]].
        Chi phí của giải pháp chỉ bao gồm các đơn hàng trong các phương tiện hợp lệ.
        """
        vehicle_loads = np.zeros(self.K)
        items_in_vehicle = [[] for _ in range(self.K)]
        actual_cost = 0 # Start with Python int
        valid_assignment = [-1] * self.N # Gán cuối cùng hợp lệ

        # Tính tổng tải trọng cho mỗi xe từ assignment_candidate
        for item_idx, vehicle_idx in enumerate(assignment_candidate):
            if vehicle_idx != -1:
                vehicle_loads[vehicle_idx] += self.demands[item_idx]
                items_in_vehicle[vehicle_idx].append(item_idx)

        # Xác thực từng xe và tính chi phí
        for k in range(self.K):
            # Rất quan trọng: Xe phải có hàng VÀ tải trọng phải nằm trong [c1, c2]
            if vehicle_loads[k] > 0 and self.c1[k] <= vehicle_loads[k] <= self.c2[k]:
                # Xe này hợp lệ, cộng chi phí của các đơn hàng trong xe này
                for item_idx in items_in_vehicle[k]:
                    actual_cost += self.costs[item_idx] # actual_cost might become a NumPy type here
                    valid_assignment[item_idx] = k # Ghi nhận gán hợp lệ
            # else: Các đơn hàng trong xe k (nếu có) không được tính vào chi phí
            # và valid_assignment[item_idx] của chúng vẫn là -1.

        return valid_assignment, actual_cost, vehicle_loads


    def solve(self):
        """Giải bài toán bằng ACO."""
        best_solution_overall = None # Lưu trữ gán hợp lệ tốt nhất
        best_cost_overall = 0      # Lưu trữ chi phí tốt nhất tương ứng (Python int initially)

        start_time = time.time()

        # Khởi tạo tau_max, tau_min cho MMAS nếu được kích hoạt
        if self.apply_mmas:
            # Ước lượng chi phí ban đầu (ví dụ: giải pháp tham lam đơn giản) để đặt tau_max
            # Hoặc một giá trị dựa trên Q và evaporation
            initial_guess_cost = np.sum(self.costs) # một ước lượng rất thô
            if initial_guess_cost == 0: initial_guess_cost = 1.0 # Avoid division by zero
            self.tau_max = self.Q / (1.0 - self.evaporation) / initial_guess_cost
            self.tau_min = self.tau_max * self.tau_min_factor
            self.pheromones.fill(self.tau_max) # Khởi tạo pheromone cho MMAS


        for iteration in range(self.num_iterations):
            if time.time() - start_time > self.time_limit:
                # if self.verbose:
                #     print(f"Time limit ({self.time_limit}s) exceeded at iteration {iteration}. Stopping.")
                break

            iteration_solutions = [] # Danh sách (valid_assignment, actual_cost, vehicle_loads)

            for ant_idx in range(self.num_ants):
                # Giai đoạn kiến xây dựng giải pháp (candidate assignment)
                current_assignment_candidate = [-1] * self.N
                current_vehicle_loads = np.zeros(self.K) # Tải trọng tạm thời khi kiến xây dựng

                order_indices = list(range(self.N))
                random.shuffle(order_indices) # Xáo trộn thứ tự đơn hàng

                for item_idx in order_indices:
                    # Lấy hàng pheromone và heuristic tương ứng với đơn hàng item_idx
                    pheromone_row = self.pheromones[item_idx, :]
                    heuristic_row = self.heuristic[item_idx, :]

                    # Chọn xe cho đơn hàng item_idx
                    selected_k = self._select_vehicle(item_idx, pheromone_row, heuristic_row, current_vehicle_loads)

                    if selected_k != -1:
                        # Gán tạm thời, chưa kiểm tra c1
                        current_assignment_candidate[item_idx] = selected_k
                        current_vehicle_loads[selected_k] += self.demands[item_idx]

                # Giai đoạn xác thực giải pháp và tính chi phí thực tế
                # Đây là điểm cải tiến QUAN TRỌNG để xử lý đúng ràng buộc c1
                valid_assignment, actual_cost, final_vehicle_loads = \
                    self._validate_and_calculate_cost(current_assignment_candidate)

                iteration_solutions.append((valid_assignment, actual_cost, final_vehicle_loads))

                # Cập nhật giải pháp tốt nhất toàn cục dựa trên chi phí thực tế
                if actual_cost > best_cost_overall:
                    best_cost_overall = actual_cost # best_cost_overall can become a NumPy type here
                    best_solution_overall = valid_assignment.copy()
                    # if self.verbose:
                    #     # Ensure printed cost is standard float for consistent display
                    #     print(f"Iter {iteration}, Ant {ant_idx}: New best cost = {float(best_cost_overall):.2f}")

                    # Cập nhật tau_max cho MMAS nếu giải pháp tốt hơn được tìm thấy (tùy chọn)
                    if self.apply_mmas and best_cost_overall > 0: # Check best_cost_overall > 0
                        self.tau_max = self.Q / (1.0 - self.evaporation) / best_cost_overall
                        self.tau_min = self.tau_max * self.tau_min_factor


            # Sắp xếp các giải pháp trong thế hệ theo chi phí thực tế giảm dần
            iteration_solutions.sort(key=lambda x: x[1], reverse=True)

            # Cập nhật Pheromone
            self.pheromones *= (1 - self.evaporation) # Bay hơi

            # Chỉ kiến tinh hoa (elite ants) được cập nhật pheromone
            # Hoặc nếu dùng MMAS, chỉ kiến tốt nhất toàn cục hoặc kiến tốt nhất thế hệ cập nhật
            ants_for_update = iteration_solutions[:self.elite_ants]
            if self.apply_mmas and best_solution_overall is not None:
                 # Trong MMAS, thường chỉ kiến tốt nhất (toàn cục hoặc thế hệ) cập nhật
                 # Ở đây, ta có thể dùng best_solution_overall để cập nhật
                 # Hoặc kiến tốt nhất của thế hệ hiện tại nếu nó cải thiện best_solution_overall
                 if iteration_solutions: # Check if iteration_solutions is not empty
                    best_iter_solution, best_iter_cost, _ = iteration_solutions[0]
                    if best_iter_cost >= best_cost_overall: # >= để xử lý trường hợp bằng nhau
                        ants_for_update = [(best_iter_solution, best_iter_cost, _)]
                    else: # Nếu không cải thiện, vẫn có thể dùng best_solution_overall để gia cố
                        ants_for_update = [(best_solution_overall, best_cost_overall, None)]
                 elif best_cost_overall > 0 : # Fallback if iteration_solutions is empty but we have a global best
                     ants_for_update = [(best_solution_overall, best_cost_overall, None)]
                 else:
                     ants_for_update = []


            for sol_assignment, sol_cost, _ in ants_for_update:
                if sol_cost > 0: # Chỉ cập nhật nếu giải pháp có chi phí dương
                    for item_idx, vehicle_idx in enumerate(sol_assignment):
                        if vehicle_idx != -1: # Chỉ cho các gán hợp lệ
                            # Lượng pheromone bồi đắp có thể tỷ lệ với chất lượng giải pháp
                            delta_pheromone = self.Q * (sol_cost / best_cost_overall if best_cost_overall > 0 else 1.0)
                            # Hoặc đơn giản là self.Q / sol_cost nếu sol_cost là hàm mục tiêu cần minimize
                            # Hoặc self.Q * sol_cost nếu sol_cost là hàm mục tiêu cần maximize (nhưng cần chuẩn hóa)
                            # Ở đây, dùng tỷ lệ với best_cost_overall để chuẩn hóa một chút
                            self.pheromones[item_idx, vehicle_idx] += delta_pheromone

            # Áp dụng giới hạn Max-Min cho Pheromone (MMAS)
            if self.apply_mmas and self.tau_min is not None and self.tau_max is not None:
                self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)
            else: 
                 self.pheromones[self.pheromones < 1e-5] = 1e-5


        elapsed_time = time.time() - start_time
        # Convert NumPy types to Python types for JSON serialization before returning
        py_best_cost_overall = 0 # Default to Python int 0

        if isinstance(best_cost_overall, np.number): # Check if it's any NumPy numeric type
            py_best_cost_overall = best_cost_overall.item() # Convert to Python native type
        elif isinstance(best_cost_overall, (int, float)): # Already a Python type
            py_best_cost_overall = best_cost_overall
        # If best_cost_overall was 0 (Python int) and remained so, it's handled.
        # If it became a NumPy type (e.g. np.int32(0)), .item() converts it to Python int 0.


        if best_solution_overall is None:
            # if self.verbose:
            #     print("No feasible solution found that satisfies all constraints (including c1).")
            # py_best_cost_overall will be 0 if best_cost_overall was 0 or np.int32(0) etc.
            return [], py_best_cost_overall, list(range(1, self.N + 1))

        final_assignments = []
        not_assigned_items = []
        for i, k_assigned in enumerate(best_solution_overall):
            if k_assigned != -1:
                final_assignments.append((int(i + 1), int(k_assigned + 1))) # Ensure Python ints
            else:
                not_assigned_items.append(int(i + 1)) # Ensure Python ints

        # if self.verbose:
        #     print(f"Final best total cost = {py_best_cost_overall:.2f}")
        #     print(f"Number of assigned items: {len(final_assignments)}")
            # print(f"Assignments: {final_assignments}")
            # print(f"Not assigned items: {not_assigned_items}")

        return final_assignments, py_best_cost_overall, not_assigned_items