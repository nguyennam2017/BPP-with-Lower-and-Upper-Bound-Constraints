import time
from utils.io_utils import read_input
def evaluate_solution(assignments, costs):
    total_cost = 0
    for order, _ in assignments:
        total_cost += costs[order - 1]  # chuyển về 0-indexing
    return total_cost

def compare_multiple_solvers(solvers, test_case):
    """
    So sánh hiệu năng và chất lượng lời giải của nhiều solver.

    Parameters:
      - solvers: list các tuple (solver_name, solver_function)
                 mỗi solver_function nhận đầu vào (N, K, demands, costs, c1, c2)
      - test_case: tuple (N, K, demands, costs, c1, c2)
      - costs: danh sách cost của các đơn hàng (dùng để tính tổng cost nếu cần)
    
    Trả về:
      Một dict kết quả với key là tên của solver và value là dict chứa:
         - time: thời gian chạy
         - total_cost: tổng cost của lời giải
         - assignments: lời giải (danh sách tuple)
    """
    N, K, demands, costs, c1, c2 = read_input(test_case)
    results = {}
    for solver_name, solver_func in solvers:
        start = time.time()
        assignments, total_cost, _ = solver_func(N, K, demands, costs, c1, c2)
        elapsed_time = time.time() - start
        
        results[solver_name] = {
            'time': elapsed_time,
            'total_cost': total_cost,
            'assignments': assignments
        }
    return results

def print_comparison(results):
    """
    In ra so sánh hiệu năng và chất lượng của các solver.
    """
    print("So sánh giữa các solver:")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Thời gian chạy: {data['time']:.6f} giây")
        print(f"  Tổng cost: {data['total_cost']}")
        print(f"  Số lượng đơn hàng phục vụ: {len(data['assignments'])}")
    print("\n---")
    
    # Tìm solver cho lời giải tốt nhất theo tổng cost (nếu có nhiều solver cùng điểm cao, liệt kê hết)
    best_cost = max(data['total_cost'] for data in results.values())
    best_solvers = [name for name, data in results.items() if data['total_cost'] == best_cost]
    
    if len(best_solvers) == 1:
        print(f"Solver '{best_solvers[0]}' có tổng cost cao nhất.")
    else:
        print("Các solver có cùng tổng cost cao nhất:", ", ".join(best_solvers))
