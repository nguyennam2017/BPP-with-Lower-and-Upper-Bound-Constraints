import random, math, copy
from typing import List

class Order:
    def __init__(self, i, quantity, cost):
        self.i = i
        self.quantity = quantity
        self.cost = cost
        self.vehicle = -1
    
    def index(self):
        return self.i + 1

    def vehicle_index(self):
        return self.vehicle + 1

class Vehicle:
    def __init__(self, i, low, hight):
        self.i = i
        self.low = low
        self.hight = hight
        self.loaded = 0
        self.collection = set()
        self.index = 0
    
    def possible(self, quantity):
        if self.loaded + quantity <= self.hight:
            return True
        return False
    
    def feasible(self):
        return self.loaded>=self.low and self.loaded<=self.hight
    
    def add(self, order: Order):
        order.vehicle = self.i
        self.collection.add(order.i)
        self.loaded += order.quantity
    
    def remove(self, order: Order):
        order.vehicle = -1
        self.collection.remove(order.i)
        self.loaded -= order.quantity
    
    def clear(self):
        self.collection.clear()
        self.loaded = 0

    def index(self):
        return self.i + 1
    

orders: List[Order] = []
vehicles: List[Vehicle] = []     
N, K = map(int, input().split())
for _ in range(N):
    i = _ 
    quantity, cost = map(int, input().split())
    orders.append(Order(i, quantity, cost))

for _ in range(K):
    i = _ 
    low, hight = map(int, input().split())
    vehicles.append(Vehicle(i, low, hight))

orders = sorted(orders, key=lambda x: x.cost/x.quantity, reverse=True)
mapping: List[int] = [0]*(len(orders)) 
# mapping anh xa thu tu order den vi tri cua order
for i, order in enumerate(orders):
    mapping[order.i] = i

def init(orders: List[Order], vehicles: List[Vehicle]):
    for order in orders:
        for vehicle in vehicles:
            if vehicle.possible(order.quantity):
                vehicle.add(order)
                break
    for vehicle in vehicles:
        if not vehicle.feasible() and vehicle.loaded !=0:
            for ord_idx in vehicle.collection:
                orders[mapping[ord_idx]].vehicle = -1
            vehicle.clear()


def compute_cost(orders: List[Order], vehicles: List[Vehicle]):
    total = 0
    for vehicle in vehicles:
        if vehicle.feasible():
            for idx in vehicle.collection:
                total += orders[mapping[idx]].cost
    return total

def simulated_annealing(orders: List[Order], vehicles: List[Vehicle], initial_temp: float = 1e10, cooling_rate: float = 0.995, epsilon: float = 1e-3):
    n, k = len(orders), len(vehicles)

    best_orders = copy.deepcopy(orders)
    best_vehicles = copy.deepcopy(vehicles)
    best_cost = compute_cost(orders ,vehicles)

    current_orders = copy.deepcopy(orders)
    current_vehicles = copy.deepcopy(vehicles)
    current_cost = best_cost

    temp = initial_temp
    while temp > epsilon:
        order_id = random.randint(0, n-1)
        vehicle_id = random.randint(-1, k-1)
        order = current_orders[mapping[order_id]]
        if order.vehicle != vehicle_id:
            if vehicle_id != -1:
                if order.vehicle == -1:
                    current_vehicles[vehicle_id].add(order)
                    new_cost = compute_cost(current_orders, current_vehicles)
                    if new_cost < current_cost:
                        prob = random.random()
                        if prob > math.exp((new_cost - current_cost)/temp):
                            current_vehicles[vehicle_id].remove(order)
                else:
                    old_vehicle_id = order.vehicle
                    current_vehicles[old_vehicle_id].remove(order)
                    current_vehicles[vehicle_id].add(order)
                    new_cost = compute_cost(current_orders, current_vehicles)
                    if new_cost < current_cost:
                        prob = random.random()
                        if prob > math.exp((new_cost - current_cost)/temp):
                            current_vehicles[vehicle_id].remove(order)
                            current_vehicles[old_vehicle_id].add(order)
            else:
                old_vehicle = current_vehicles[order.vehicle]
                old_vehicle.remove(order)
                new_cost = compute_cost(current_orders, current_vehicles)
                if new_cost < current_cost:
                    prob = random.random()
                    if prob > math.exp((new_cost - current_cost)/temp):
                        old_vehicle.add(order)

        current_cost = compute_cost(current_orders, current_vehicles)
        if current_cost > best_cost:
            best_orders = copy.deepcopy(current_orders)
            best_vehicles = copy.deepcopy(current_vehicles)
            
        temp *= cooling_rate

    return best_orders, best_vehicles


init(orders=orders, vehicles=vehicles)
orders, vehicles = simulated_annealing(orders, vehicles)

count = 0
for vehicle in vehicles:
    if vehicle.feasible():
        count += len(vehicle.collection)

print(count)
for vehicle in vehicles:
    if vehicle.feasible():
        for idx in vehicle.collection:
            order = orders[mapping[idx]]
            print(f'{order.index()} {order.vehicle_index()}')
