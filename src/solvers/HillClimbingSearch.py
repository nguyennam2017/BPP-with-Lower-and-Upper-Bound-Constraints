import random
class HillClimbingSearch():
    def __init__(self, N, K, demands, costs, c1, c2,
                 num_iterations= 10000):
        self.N = N
        self.K = K
        self.d = demands
        self.c = costs
        self.c1 = [0] + c1
        self.c2 = [0] + c2
        self.num_iterations = num_iterations
        self.load = [0 for j in range(self.K + 1)]
        self.cost = [0 for j in range(self.K + 1)]
        self.x = [0 for i in range(self.N)]
        self.obj = 0
        self.best_obj = 0
        self.best_X = self.x[:]
        self.best_load = self.load[:]
    
    def init_solution(self):
        d = self.d
        c = self.c
        c1 = self.c1
        c2 = self.c2
        load = self.load
        cost = self.cost
        x = self.x
        order_indices = list(range(self.N))
        order_indices.sort(key=lambda i: c[i] / d[i], reverse=True)
        
        for i in order_indices:
            for j in range(1, self.K + 1):
                if load[j] + d[i] <= c2[j]:  
                    x[i] = j
                    load[j] += d[i]
                    cost[j] += c[i]
                    break

        for j in range(1, self.K + 1):
            if load[j] >= c1[j] and load[j] <= c2[j]:
                self.obj += cost[j]

        self.cost = cost[:]
        self.load = load[:]
        self.x = x[:]
        self.best_obj = self.obj
        self.best_x = x[:]
        self.best_load = load[:]

    def get_cur_update(self, order, cur_vehicle, vehicle):
        load = self.load
        cost = self.cost
        d = self.d
        c = self.c
        c1 = self.c1
        c2 = self.c2
        delta_cost = 0
        if cur_vehicle != 0:
            before_load_cur = load[cur_vehicle]
            after_load_cur = before_load_cur - d[order]
            if before_load_cur > c2[cur_vehicle]: # Trước đó xe quá trọng tải
                if after_load_cur > c2[cur_vehicle]: # Vẫn quá trọng tải
                    delta_cost -= 0
                elif after_load_cur >= c1[cur_vehicle]: # Trở nên khả thi
                    delta_cost += cost[cur_vehicle] - c[order]
                else: # Trở nên thiếu cân
                    delta_cost -= 0
            elif before_load_cur >= c1[cur_vehicle]: # Trước đó xe thỏa mãn
                if after_load_cur >= c1[cur_vehicle]: # Vẫn thỏa mãn
                    delta_cost -= c[order]
                else: # Trở nên thiếu cân
                    delta_cost -= cost[cur_vehicle]
            else: # Trước đó xe thiếu cân
                delta_cost -= 0
        if vehicle == 0:
            return delta_cost
        else:
            before_load_vehicle = load[vehicle]
            after_load_vehicle = before_load_vehicle + d[order]
            if before_load_vehicle > c2[vehicle]: # Trước đó xe quá tải trọng
                delta_cost += 0 # Không thay đổi được gì
            elif before_load_vehicle >= c1[vehicle]: # Trước đó xe thỏa mãn
                if after_load_vehicle <= c2[vehicle]: # Vẫn thỏa mãn
                    delta_cost += c[order]
                else: # Quá tải trọng
                    delta_cost -= cost[vehicle]
            else: # Trước đó xe thiếu cân
                if after_load_vehicle < c1[vehicle]: # Vẫn thiếu cân
                    delta_cost += 0
                elif after_load_vehicle <= c2[vehicle]: # Thỏa mãn
                    delta_cost += c[order] + cost[vehicle]
                else: # Quá tải trọng
                    delta_cost += 0
        return delta_cost
    
    def update(self):
        local_maxima = 0
        flat_point = 0
        limit = self.N/2
        for it in range(self.num_iterations):
            order = random.randint(0, self.N - 1)
            cur_vehicle = self.x[order]
            cand = []
            best_update = 0
            for vehicle in range(0, self.K + 1):
                if vehicle != cur_vehicle:
                    cur_update = self.get_cur_update(order, cur_vehicle, vehicle)
                    if cur_update > best_update:
                        best_update = cur_update
                        cand.clear()
                        cand.append(vehicle)
                    elif cur_update == best_update:
                        cand.append(vehicle)
            if not cand:
                local_maxima += 1
                if local_maxima == limit:
                    new_vehicle = random.randint(1, self.K)
                    best_update = self.get_cur_update(order, cur_vehicle, new_vehicle)
                    local_maxima = 0
                    flat_point = 0
                else:
                    continue
            else:
                if best_update == 0:
                    flat_point += 1
                    if flat_point == limit:
                        new_vehicle = random.choice(cand)
                        flat_point = 0
                        local_maxima = 0
                    else:
                        continue
                else:
                    new_vehicle = random.choice(cand)
                    local_maxima = 0
                    flat_point = 0
            # Update current
            self.obj += best_update
            self.x[order] = new_vehicle
            if cur_vehicle != 0:
                self.load[cur_vehicle] -= self.d[order]
                self.cost[cur_vehicle] -= self.c[order]
            if new_vehicle != 0:
                self.load[new_vehicle] += self.d[order]
                self.cost[new_vehicle] += self.c[order]
            # Update best
            if self.obj + best_update > self.best_obj:
                self.best_obj = self.obj + best_update
                self.best_x = self.x[:]
                self.best_load = self.load[:]

    def print_solution(self):
        best_x = self.best_x
        best_load = self.best_load
        c1 = self.c1
        c2 = self.c2
        for vehicle in range(1, self.K + 1):
            orders_in_vehicle = [order for order in range(self.N) if best_x[order] == vehicle]
            if best_load[vehicle] > c2[vehicle] or best_load[vehicle] < c1[vehicle]:
                for order in orders_in_vehicle:
                    best_x[order] = 0
        served = 0
        for order in range(self.N):
            if best_x[order] != 0:
                served += 1
        print(served)
        for order in range(self.N):
            if best_x[order] != 0:
                print(f"{order + 1} {best_x[order]}")        

    def solve(self):
        self.init_solution()
        self.update()
        self.print_solution()


# n, k = list(map(int, input().split()))
# d = []
# c = []
# c1 = []
# c2 = []
# for i in range(n):
#     di, ci = list(map(int, input().split()))
#     d.append(di)
#     c.append(ci)
# for i in range(k):
#     c1i, c2i = list(map(int, input().split()))
#     c1.append(c1i)
#     c2.append(c2i)

# solver = HillClimbingSearch(n, k, d, c, c1, c2)
# solver.solve()