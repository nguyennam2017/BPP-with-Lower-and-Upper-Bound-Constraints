
from ortools.linear_solver import pywraplp
import time
def lp_solve_binpacking(N, K, D, C, c1, c2,time_limit=float('inf')):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # x[i][k] = 1 nếu đơn hàng i gán cho xe k
    x = {}
    for i in range(N):
        for k in range(K):
            x[i, k] = solver.BoolVar(f'x_{i}_{k}')
    # y[k] = 1 nếu xe k được dùng
    y = [solver.BoolVar(f'y_{k}') for k in range(K)]
    # Mỗi đơn hàng chỉ được gán cho 1 xe
    for i in range(N):
        solver.Add(solver.Sum(x[i, k] for k in range(K)) <= 1)
    # Tải của mỗi xe trong giới hạn [c1(k), c2(k)] nếu được dùng
    for k in range(K):
        total_load = solver.Sum(D[i] * x[i, k] for i in range(N))
        solver.Add(total_load >= c1[k] )
        solver.Add(total_load <= c2[k])
    # Hàm mục tiêu: tối đa hóa tổng chi phí các đơn hàng được phục vụ
    solver.Maximize(
        solver.Sum(C[i] * x[i, k] for i in range(N) for k in range(K))
    )
    solver.max_time_in_seconds = time_limit
    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time

    result = []
    total_cost = 0
    not_assigned = []
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        
        for i in range(N):
            assigned = False
            for k in range(K):
                if x[i, k].solution_value() > 0.5:
                    result.append((i + 1, k + 1))  # đánh số từ 1
                    total_cost += C[i]
                    assigned = True
            if not assigned:
                not_assigned.append(i + 1)
    # else:
    #     # Nếu không tìm được giải pháp (vượt thời gian hoặc không khả thi)
    #     print(f"No solution found within time limit of {time_limit}")
    
    return result, total_cost, not_assigned

