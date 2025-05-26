from solvers.cp_solver import cp_solve_binpacking
from solvers.ACO_solver import ACOBinPackingSolver
from solvers.GA_solver import GABinPackingSolver
from solvers.HillClimbingSearch import HillClimbingSearch
from solvers.SA_solver import SimulatedAnnealingSearch
from utils.performance import benchmark_solver
def main():
    Solvers = {
        "CP-SAT": cp_solve_binpacking,
        "ACO"   : ACOBinPackingSolver,
        "GA"    : GABinPackingSolver,
        "HillClimbing" : HillClimbingSearch,
        "SA"    : SimulatedAnnealingSearch
    }
    Category = ["small",    # Tescase nhỏ, có thể tính chính xác
                "edge",     # Trường hợp biên
                "stress",   # Trường hợp N,K lớn
                "random",   # Trường hợp random
            ]   
    #benchmark_solver(solver_name= "ACO", solver_func = ACOBinPackingSolver, category = "hustack", num_runs=10)
    for category in Category:
        # for solver_name, solver_func in Solvers.items():
        #     benchmark_solver(solver_name=solver_name, solver_func=solver_func, category=category, num_runs=1, time_limit=10)
        benchmark_solver(solver_name="SA", solver_func=SimulatedAnnealingSearch, category=category, num_runs=10, time_limit=10)
if __name__ == "__main__":
    main()
