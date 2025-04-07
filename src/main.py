from solvers.cp_solver import cp_solve_binpacking
from solvers.ACO_solver import ACOBinPackingSolver
from utils.performance import benchmark_solver
def main():
    Solvers = {
        "CP-SAT": cp_solve_binpacking,
        "ACO"   : ACOBinPackingSolver,
    }
    Category = ["small",    # Tescase nhỏ, có thể tính chính xác
                "edge",     # Trường hợp biên
                "stress",   # Trường hợp N,K lớn
                "random",   # Trường hợp random
                "hustack"   # Các test trên hustack
            ]   
    #benchmark_solver(solver_name= "ACO", solver_func = ACOBinPackingSolver, category = "hustack", num_runs=10)
    for category in Category:
        benchmark_solver(solver_name = "CP-SAT", solver_func=cp_solve_binpacking, category=category, num_runs=1, time_limit=5)
if __name__ == "__main__":
    main()
