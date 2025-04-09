from ortools.sat.python import cp_model
import time

def cp_solve_binpacking(N, K, demands, costs, c1, c2, time_limit=float('inf')):
    """
    Giải bài toán Bin Packing với ràng buộc dưới và trên
    sử dụng Google OR-Tools (CP-SAT), dừng lại nếu vượt quá time_limit.

    Mục tiêu: Tối đa hóa tổng cost của các đơn hàng được phục vụ.
    Ràng buộc:
      - Mỗi đơn hàng được gán tối đa cho 1 phương tiện.
      - Với mỗi phương tiện k, nếu được sử dụng, tổng demand phải trong khoảng [c1[k], c2[k]].
    
    Args:
        N (int): Số đơn hàng
        K (int): Số phương tiện
        demands (list): Danh sách khối lượng của các đơn hàng
        costs (list): Danh sách chi phí của các đơn hàng
        c1 (list): Danh sách giới hạn dưới của các phương tiện
        c2 (list): Danh sách giới hạn trên của các phương tiện
        time_limit (float): Giới hạn thời gian giải (giây), mặc định là vô cực

    Returns:
        tuple: (assignments, total_cost, not_assigned)
            - assignments: Danh sách (order_id, vehicle_id) của các đơn hàng được gán
            - total_cost: Tổng chi phí của các đơn hàng được gán
            - not_assigned: Danh sách các đơn hàng không được gán
    """
    # Khởi tạo model CP-SAT
    model = cp_model.CpModel()

    # Biến nhị phân x[i][k] = 1 nếu đơn hàng i được phục vụ bởi xe k
    x = []
    for i in range(N):
        x.append([model.NewBoolVar(f'x_{i}_{k}') for k in range(K)])
    
    # Biến nhị phân y[k] = 1 nếu xe k được sử dụng
    y = [model.NewBoolVar(f'y_{k}') for k in range(K)]
    
    # Ràng buộc: Mỗi đơn hàng chỉ được phục vụ bởi tối đa 1 xe
    for i in range(N):
        model.Add(sum(x[i][k] for k in range(K)) <= 1)
    
    # Ràng buộc: Với mỗi xe k, tổng demand phải trong khoảng [c1[k], c2[k]] nếu xe k được dùng
    for k in range(K):
        total_demand_k = sum(demands[i] * x[i][k] for i in range(N))
        model.Add(total_demand_k >= c1[k] * y[k])
        model.Add(total_demand_k <= c2[k] * y[k])
    
    # Ràng buộc phụ: Nếu một đơn hàng được gán cho xe k thì xe k phải được sử dụng
    for i in range(N):
        for k in range(K):
            model.Add(x[i][k] <= y[k])
    
    # Hàm mục tiêu: Tối đa hóa tổng cost của các đơn hàng được phục vụ
    model.Maximize(sum(costs[i] * x[i][k] for i in range(N) for k in range(K)))
    
    # Khởi tạo solver
    solver = cp_model.CpSolver()
    # Thiết lập giới hạn thời gian (tính bằng giây)
    solver.parameters.max_time_in_seconds = time_limit
    
    # Giải bài toán
    start_time = time.time()
    status = solver.Solve(model)
    elapsed_time = time.time() - start_time
    
    # Khởi tạo kết quả mặc định
    assignments = []
    total_cost = 0
    not_assigned = list(range(1, N + 1))  # Ban đầu giả sử tất cả đều không được gán

    # Xử lý kết quả
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Nếu tìm được giải pháp (tối ưu hoặc khả thi)
        assignments = []
        total_cost = 0
        not_assigned = []
        for i in range(N):
            assigned = False
            for k in range(K):
                if solver.Value(x[i][k]) == 1:
                    assignments.append((i + 1, k + 1))  # (đơn hàng, xe)
                    total_cost += costs[i]
                    assigned = True
            if not assigned:
                not_assigned.append(i + 1)
    else:
        # Nếu không tìm được giải pháp (vượt thời gian hoặc không khả thi)
        print(f"No solution found within time limit of {time_limit}s. Status: {solver.StatusName(status)}")

    return assignments, total_cost, not_assigned