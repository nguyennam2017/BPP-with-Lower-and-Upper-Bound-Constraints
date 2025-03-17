# Bin Packing with Lower and Upper Bound Constraints

## Problem Statement

- There are **N** orders **(1, 2, ..., N)**, in which order **i** has quantity **d(i)** and cost **c(i)**.
- There are **K** vehicles **(1, 2, ..., K)** for serving orders, where vehicle **k** has a lower-bound capacity **c1(k)** and an upper-bound capacity **c2(k)**.  
  Compute a solution for assigning orders to vehicles such that:
  - Each order is served by at most one vehicle.
  - The sum of the quantities of orders loaded (served) in a vehicle must be between the lower and upper capacity of that vehicle.
  - The total cost of served orders is maximized.

## Input

- **Line 1**: Contains two positive integers **N** and **K**  
  (**1 ≤ N ≤ 1000, 1 ≤ K ≤ 100**)
- **Line (i+1)** (for **i = 1 to N**): Contains two integers **d(i)** and **c(i)**  
  (**1 ≤ d(i), c(i) ≤ 100**)
- **Line (N+1+k)** (for **k = 1 to K**): Contains two integers **c1(k)** and **c2(k)**  
  (**1 ≤ c1(k) ≤ c2(k) ≤ 1000**)

## Output

The output should contain the optimal assignment of orders to vehicles while maximizing the total cost.

### Format:
- **Line 1**: An integer **m**, the number of assigned orders.
- **Next m lines**: Each line contains two integers **i** and **b**,  
  where order **i** is served by vehicle **b**.



## Directory structure
```
project-root/
│── data/
│   ├── input/        # Directory containing input files
│   ├── output/       # Directory containing output results
│
│── src/
│   ├── solvers/      # Contains bin packing solving algorithms
│   │   ├── cp_solver.py
│   │
│   ├── utils/        # Utilities for handling I/O
│   │   ├── io_utils.py
│   │
│   ├── main.py       # Main program
│
│── README.md         # User guide
│── requirements.txt  # List of required libraries
```

## Installation

   ```sh
   pip install -r requirements.txt
   ```

## Usage
```sh
python src/main.py
```
The results will be saved in the `data/output/` directory.

## Team Members

- 20224942    Vũ Đình Cường
- 20224939    Nguyễn Kim Cường
- 20224992    Trần Mạnh Hoàng
- 20220037    Nguyễn Nam
- 20225033    Nguyễn Duy Lợi

