import sys
from solvers.cp_solver import solve_binpacking
from utils.performance import compare_multiple_solvers
from utils.performance import print_comparison
def main():
    test_case = "data/input/sample_input.txt"    

    Solvers = [("CP_SAT", solve_binpacking)]
    results = compare_multiple_solvers(solvers = Solvers, test_case = test_case)
    print_comparison(results=results)
if __name__ == "__main__":
    main()
