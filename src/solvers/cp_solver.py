from ortools.sat.python import cp_model

def solve_binpacking(N, K, demands, costs, c1, c2):
    """
    Giải bài toán Bin Packing với ràng buộc dưới và trên
    sử dụng Google OR-Tools (CP-SAT).

    Mục tiêu: Tối đa hóa tổng cost của các đơn hàng được phục vụ.
    Ràng buộc:
      - Mỗi đơn hàng được gán tối đa cho 1 phương tiện.
      - Với mỗi phương tiện k, nếu được sử dụng, tổng demand phải trong khoảng [c1[k], c2[k]].
    """
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
    
    # Giải bài toán
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    assignments = []
    total_cost = 0
    not_assigned = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            assigned = False
            for k in range(K):
                if solver.Value(x[i][k]) == 1:
                    assignments.append((i + 1, k + 1))  # (đơn hàng, xe)
                    total_cost += costs[i]
                    assigned = True
            if not assigned:
                not_assigned.append(i + 1)
        
        print(f"Tổng cost tối ưu: {total_cost}")
    
    return assignments, total_cost, not_assigned
