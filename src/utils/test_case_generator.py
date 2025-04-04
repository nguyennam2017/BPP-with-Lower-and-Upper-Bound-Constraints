# File: src/utils/test_case_generator.py
import random
import os

def generate_test_case(file_path, N=10, K=3,
                                order_min=1, order_max=100,
                                vehicle_min=1, vehicle_max=1000,
                                seed=None):
    """
    Sinh dữ liệu test tự động cho bài toán phân bổ đơn hàng cho xe và lưu ra file txt.

    Parameters:
      - file_path: đường dẫn đến file cần lưu (ví dụ: "data/input/test_case1.txt")
      - N: số lượng đơn hàng (mặc định 10)
      - K: số lượng xe (mặc định 3)
      - order_min, order_max: khoảng giá trị cho demand và cost của đơn hàng (mặc định từ 1 đến 100)
      - vehicle_min, vehicle_max: khoảng giá trị cho low-bound và up-capacity của xe (mặc định từ 1 đến 1000)
      - seed: seed cho bộ sinh số ngẫu nhiên (nếu cần tái lập lại kết quả)

    Hàm sẽ sinh ra dữ liệu theo định dạng:
      Line 1: N K
      Next N lines: d(i) c(i)
      Next K lines: c1(k) c2(k)
    Và lưu dữ liệu vào file được chỉ định.
    """
    if seed is not None:
        random.seed(seed)
    
    # Sinh dữ liệu cho đơn hàng
    orders = []
    for _ in range(N):
        d = random.randint(order_min, order_max)
        c = random.randint(order_min, order_max)
        orders.append((d, c))
    
    # Sinh dữ liệu cho xe, đảm bảo c1 <= c2
    vehicles = []
    for _ in range(K):
        c1 = random.randint(vehicle_min, vehicle_max)
        c2 = random.randint(c1, vehicle_max)
        vehicles.append((c1, c2))
    
    # Tạo chuỗi dữ liệu theo định dạng yêu cầu:
    #   Line 1: N K
    #   Next N lines: d(i) c(i)
    #   Next K lines: c1(k) c2(k)
    lines = []
    lines.append(f"{N} {K}")
    for d, c in orders:
        lines.append(f"{d} {c}")
    for c1, c2 in vehicles:
        lines.append(f"{c1} {c2}")
    
    test_case_str = "\n".join(lines)
    
    # Tạo thư mục chứa file nếu chưa tồn tại
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(test_case_str)
    
    print(f"Test case đã được sinh và lưu vào file: {file_path}")
    return test_case_str


