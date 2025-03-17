import sys
from utils.io_utils import read_input, write_output
from solvers.cp_solver import solve_binpacking

def main():
    input_file = "data/input/sample_input.txt"
    output_file = "data/output/sample_output.txt"

    N, K, demands, costs, c1, c2 = read_input(input_file)

    assignments, total_cost, not_assigned = solve_binpacking(N, K, demands, costs, c1, c2)

    write_output(assignments, total_cost, not_assigned, output_file)
    print(f"Kết quả đã được ghi ra file: {output_file}")

if __name__ == "__main__":
    main()
