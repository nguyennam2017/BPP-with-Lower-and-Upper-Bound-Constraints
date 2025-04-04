import sys
from solvers.cp_solver import solve_binpacking
from solvers.ACO_solver import ACOBinPackingSolver

from utils.performance import compare_multiple_solvers
from utils.performance import print_comparison
from utils.test_case_generator import generate_test_case
def main():
    test_case = "data/input/sample_input.txt"    
    
    gen_test_case = "data/input/gen_input.txt"    
    generate_test_case(file_path=gen_test_case, N = 10, K= 5 )


    Solvers = [
    ("CP-SAT", solve_binpacking),
    ("ACO", ACOBinPackingSolver)
]
    results = compare_multiple_solvers(solvers = Solvers, test_case = gen_test_case)
    print_comparison(results=results)
if __name__ == "__main__":
    main()
