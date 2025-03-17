import sys
from utils.io_utils import read_input, write_output
from solvers.cp_solver import solve_binpacking

def main():
    # Đường dẫn file input và output (có thể cấu hình theo ý bạn)
    input_file = "data/input/sample_input.txt"
    output_file = "data/output/sample_output.txt"

    # Đọc input
    N, K, demands, costs, c1, c2 = read_input(input_file)

    # Giải bài toán
    assignments = solve_binpacking(N, K, demands, costs, c1, c2)

    # Ghi output
    write_output(assignments, output_file)
    print(f"Kết quả đã được ghi ra file: {output_file}")

if __name__ == "__main__":
    main()
