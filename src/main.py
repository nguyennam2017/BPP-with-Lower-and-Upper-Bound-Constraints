from solvers.cp_solver import cp_solve_binpacking
from solvers.ACO_solver import ACOBinPackingSolver
from solvers.GA_solver import GABinPackingSolver
from utils.performance import benchmark_solver

def main():
    Solvers = {
        "CP-SAT": cp_solve_binpacking,
        "ACO"   : ACOBinPackingSolver,
        "GA"    : GABinPackingSolver
    }
    Category = ["small",    # Tescase nhỏ, có thể tính chính xác
                "edge",     # Trường hợp biên
                "stress",   # Trường hợp N,K lớn
                "random",   # Trường hợp random
                "hustack"   # Các test trên hustack
            ]
    benchmark_solver(solver_name= "GA", solver_func = GABinPackingSolver, category = "small", num_runs=1)
    # for category in Category:
    #     benchmark_solver(solver_name = "GA", solver_func=GABinPackingSolver, category=category, num_runs=20, time_limit=10)
if __name__ == "__main__":
    main()
