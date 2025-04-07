# File: src/utils/performance.py

import time
import os
import json
import statistics
from utils.io_utils import read_input

def run_single_solver(solver_func, input_path, time_limit=float('inf')):
    """
    Chạy một solver (function hoặc class) trên một test case.

    Returns:
      - dict chứa:
          - assignments: danh sách (order_id, vehicle_id)
          - total_cost: tổng chi phí
          - time: thời gian chạy (float, giây)
    """
    N, K, demands, costs, c1, c2 = read_input(input_path)

    start = time.time()

    # Solver là class
    if isinstance(solver_func, type):
        solver_instance = solver_func(N=N, K=K, demands=demands, costs=costs, c1=c1, c2=c2, time_limit=time_limit)
        assignments, total_cost, _ = solver_instance.solve()
    else:
        # Solver là function
        assignments, total_cost, _ = solver_func(N, K, demands, costs, c1, c2, time_limit=time_limit)

    elapsed = round(time.time() - start, 4)

    return {
        "assignments": assignments,
        "total_cost": total_cost,
        "time": elapsed
    }


def benchmark_solver(solver_name, solver_func, category, num_runs=1, time_limit=float('inf'), input_dir="data/input", output_dir="data/output"):
    """
    Chạy solver trên toàn bộ test case của một category với số lần chạy xác định,
    ghi output và meta. Nếu num_runs = 1, chỉ ghi total_cost; nếu num_runs > 1, 
    ghi thêm min_cost, max_cost, average_cost, và std của total_cost.

    Args:
        solver_name (str): Tên của solver
        solver_func: Hàm hoặc class solver
        category (str): Danh mục test case
        num_runs (int): Số lần chạy cho mỗi test case (mặc định là 1)
        time_limit (float): Giới hạn thời gian cho mỗi lần chạy (giây)
        input_dir (str): Thư mục chứa file input
        output_dir (str): Thư mục chứa file output

    Lưu file:
      - output:   .out.txt (gán đơn hàng của lần chạy có total_cost cao nhất)
      - meta:     .meta.json (thời gian và các thông số cost tùy theo num_runs)
    """
    input_folder = os.path.join(input_dir, category)
    output_folder = os.path.join(output_dir, solver_name, category)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n Benchmarking '{solver_name}' on category '{category}' with {num_runs} runs per test case...")

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, file_name)
        output_name = file_name.replace(".txt", ".out.txt")
        meta_name = file_name.replace(".txt", ".meta.json")

        # Chạy solver num_runs lần cho mỗi test case
        results = []
        for _ in range(num_runs):
            result = run_single_solver(solver_func, input_path, time_limit=time_limit)
            results.append(result)

        # Lấy lần chạy có total_cost cao nhất để ghi file output
        best_result = max(results, key=lambda x: x["total_cost"])
        total_time = sum(r["time"] for r in results)  # Tổng thời gian của tất cả lần chạy

        # Ghi file lời giải (dựa trên lần chạy tốt nhất)
        with open(os.path.join(output_folder, output_name), 'w') as f_out:
            f_out.write(f"{len(best_result['assignments'])}\n")
            for order, vehicle in best_result['assignments']:
                f_out.write(f"{order} {vehicle}\n")

        # Xử lý metadata và in kết quả tùy theo num_runs
        if num_runs == 1:
            # Trường hợp num_runs = 1: Chỉ ghi và in total_cost
            total_cost = best_result["total_cost"]
            meta_data = {
                "input_file": file_name,
                "num_runs": num_runs,
                "total_cost": total_cost,
                "total_time": round(total_time, 4)
            }
            print(f"{file_name:<25} cost = {total_cost:<6} time = {round(total_time, 4)}s")
        else:
            # Trường hợp num_runs > 1: Tính và ghi đầy đủ thống kê
            total_costs = [r["total_cost"] for r in results]
            min_cost = min(total_costs)
            max_cost = max(total_costs)
            average_cost = round(statistics.mean(total_costs), 2)
            std_cost = round(statistics.stdev(total_costs), 2)

            meta_data = {
                "input_file": file_name,
                "num_runs": num_runs,
                "min_cost": min_cost,
                "max_cost": max_cost,
                "average_cost": average_cost,
                "std_cost": std_cost,
                "total_time": round(total_time, 4)
            }
            print(f"{file_name:<25} min_cost = {min_cost:<6} max_cost = {max_cost:<6} "
                  f"avg_cost = {average_cost:<6} std = {std_cost:<6} time = {round(total_time, 4)}s")

        # Ghi file metadata
        with open(os.path.join(output_folder, meta_name), 'w') as f_meta:
            json.dump(meta_data, f_meta, indent=2)