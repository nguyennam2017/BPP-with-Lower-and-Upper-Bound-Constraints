import sys

def read_input(file_path):
    """Đọc dữ liệu đầu vào từ file txt.
    Args:
        file_path (str): Đường dẫn đến file đầu vào chứa thông tin đơn hàng và phương tiện.
    
    Returns:
        tuple: (N, K, demands, costs, c1, c2) chứa số đơn hàng, số phương tiện, 
               danh sách khối lượng, chi phí đơn hàng, giới hạn dưới và trên của phương tiện.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except IOError:
        print(f"Error reading file: {file_path}")
        sys.exit(1)

    tokens = ' '.join([line.strip() for line in lines]).split()
    
    # Đọc N và K
    N = int(tokens[0])
    K = int(tokens[1])
    demands = []
    costs = []
    index = 2

    # Đọc N đơn hàng
    for _ in range(N):
        d_i = int(tokens[index])
        c_i = int(tokens[index + 1])
        demands.append(d_i)
        costs.append(c_i)
        index += 2
    
    # Đọc K phương tiện
    c1 = []
    c2 = []
    for _ in range(K):
        c1_k = int(tokens[index])
        c2_k = int(tokens[index + 1])
        c1.append(c1_k)
        c2.append(c2_k)
        index += 2
    
    return N, K, demands, costs, c1, c2

def write_output(assignments, total_cost, not_assigned, file_path):
    """Ghi kết quả ra file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Tổng cost tối ưu: {total_cost}\n")
            m = len(assignments)
            f.write(f"{m}\n")
            for order_id, vehicle_id in assignments:
                f.write(f"{order_id} {vehicle_id}\n")
            for i in not_assigned:
                f.write(f"Đơn hàng {i} không được vận chuyển\n")
    except IOError:
        print(f"Error writing to file: {file_path}")
        sys.exit(1)
