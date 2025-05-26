import random
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.test_config import test_case_settings

def generate_single_case(N, K,
                         order_min, order_max,
                         vehicle_min, vehicle_max):
    orders = [(random.randint(order_min, order_max),
               random.randint(order_min, order_max)) for _ in range(N)]
    
    vehicles = []
    for _ in range(K):
        c1 = random.randint(vehicle_min, vehicle_max)
        c2 = random.randint(c1, vehicle_max)
        vehicles.append((c1, c2))

    return orders, vehicles

def format_case_to_str(N, K, orders, vehicles):
    lines = [f"{N} {K}"]
    lines += [f"{d} {c}" for d, c in orders]
    lines += [f"{c1} {c2}" for c1, c2 in vehicles]
    return "\n".join(lines)

def generate_test_cases(base_dir="data/input/",
                        category="random",
                        num_cases=5,
                        N=10, K=3,
                        order_min=1, order_max=100,
                        vehicle_min=1, vehicle_max=1000,
                        seed=None):
    """
    Tạo nhiều test case tự động và lưu vào thư mục theo category.
    
    Parameters:
        - base_dir: thư mục gốc lưu file input
        - category: tên loại test case ("small", "edge", "stress", "random")
        - num_cases: số lượng test case cần sinh
        - N, K: số đơn hàng và xe mỗi test case
        - *_min, *_max: khoảng giá trị của đơn hàng và xe
        - seed: seed để tạo kết quả có thể lặp lại
    """
    if seed is not None:
        random.seed(seed)

    target_dir = os.path.join(base_dir, category)
    os.makedirs(target_dir, exist_ok=True)

    meta_log = []

    for i in range(1, num_cases + 1):
        orders, vehicles = generate_single_case(N, K, order_min, order_max, vehicle_min, vehicle_max)
        case_str = format_case_to_str(N, K, orders, vehicles)

        file_name = f"test_{category}_{i}.txt"
        file_path = os.path.join(target_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(case_str)

        meta_log.append(f"{file_name}: N={N}, K={K}, orders=[{order_min}-{order_max}], vehicles=[{vehicle_min}-{vehicle_max}]")

    # Ghi file mô tả
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(target_dir, f"meta_{timestamp}.log"), 'w') as log_file:
        log_file.write("\n".join(meta_log))

    print(f"Đã sinh {num_cases} test case vào thư mục: {target_dir}")

def run_generate_all():
    for category, config in test_case_settings.items():
        print(f"Generating {category} test cases...")
        generate_test_cases(
            base_dir="data/input",
            category=category,
            num_cases=config["num_cases"],
            N=config["N"],
            K=config["K"],
            order_min=config["order_min"],
            order_max=config["order_max"],
            vehicle_min=config["vehicle_min"],
            vehicle_max=config["vehicle_max"],
            seed=42  # Cố định
        )

if __name__ == "__main__":
    run_generate_all()
