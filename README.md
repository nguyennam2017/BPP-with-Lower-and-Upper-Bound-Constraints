# Bin Packing Solver


## Cấu trúc thư mục
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

## Cài đặt
Cài đặt các thư viện cần thiết:
   ```sh
   pip install -r requirements.txt
   ```

## Sử dụng
Chạy chương trình với lệnh:
```sh
python src/main.py
```
Kết quả sẽ được lưu trong thư mục `data/output/`.


