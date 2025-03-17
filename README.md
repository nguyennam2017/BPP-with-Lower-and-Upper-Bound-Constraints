# Bin Packing with Lower and Upper Bound Constraints
## Problem Statement
- There are $N$ orders $1, 2, \dots, N$, in which order $ i $ has quantity $ d(i) $ and cost $ c(i) $.
- There are $ K $ vehicles $ 1, 2, \dots, K $ for serving orders in which vehicle $ k $ has low-bound capacity $ c_1(k) $ and up-capacity $ c_2(k) $. Compute a solution for assigning orders to vehicles such that:
  - Each order is served by at most one vehicle.
  - Sum of quantity of orders loaded (served) in a vehicle must be between the low-capacity and up-capacity of that vehicle.
  - Total cost of served orders is maximal.
## Input
- **Line 1**: contains positive integers $ N $ và $ K $ ( $ 1 \leq N \leq 1000, 1 \leq K \leq 100 $ )
- **Line $ i + 1 $ ($ i = 1, \dots, N $)**: contains 2 integers $ d(i) $ and $ c(i) $ ( $ 1 \leq d(i), c(i) \leq 100 $ )
- **Line $ N + 1 + k $ ($ k = 1, \dots, K $)**: contains 2 integers $ c_1(k) $ and $ c_2(k) $ ( $ 1 \leq c_1(k) \leq c_2(k) \leq 1000 $ )

## Output
- **Line 1**: contains an integer $ m $
- **Line $ i + 1 $ ($ i = 1, 2, \dots, m $)**: contains 2 positive integers $ i $ and $ b $ in which order $ i $ is served by vehicle $ b $


## Directory structure
```
project-root/
│── data/
│   ├── input/        # Thư mục chứa các tệp đầu vào
│   ├── output/       # Thư mục chứa kết quả đầu ra
│
│── src/
│   ├── solvers/      # Chứa thuật toán giải bin packing
│   │   ├── cp_solver.py
│   │
│   ├── utils/        # Các tiện ích xử lý I/O
│   │   ├── io_utils.py
│   │
│   ├── main.py       # Chương trình chính
│
│── README.md         # Hướng dẫn sử dụng
│── requirements.txt  # Danh sách thư viện cần thiết
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
