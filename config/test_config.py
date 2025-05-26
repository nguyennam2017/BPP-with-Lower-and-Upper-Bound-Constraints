# File: config/test_config.py

test_case_settings = {
    "small": { # Các trường hợp nhỏ
        "num_cases": 5,
        "N": 10,
        "K": 3,
        "order_min": 1,
        "order_max": 50,
        "vehicle_min": 10,
        "vehicle_max": 100
    },
    "edge": { # Các trường hợp biên
        "num_cases": 5,
        "N": 10,
        "K": 3,
        "order_min": 1,
        "order_max": 100,
        "vehicle_min": 10,
        "vehicle_max": 15  # Biên sát nhau
    },
    "stress": { # Các trường hợp N,K lớn, để đo khả năng và tính mở rộng nếu có thể
        "num_cases": 5,
        "N": 500,
        "K": 100,
        "order_min": 50,
        "order_max": 100,
        "vehicle_min": 200,
        "vehicle_max": 1000
    },
    "random": { # Trường hợp random
        "num_cases": 5,
        "N": 50,
        "K": 10,
        "order_min": 1,
        "order_max": 100,
        "vehicle_min": 100,
        "vehicle_max": 500
    },
    "huge": { # Trường hợp random
        "num_cases": 5,
        "N": 1000,
        "K": 100,
        "order_min": 1,
        "order_max": 100,
        "vehicle_min": 1,
        "vehicle_max": 1000
    },

}
