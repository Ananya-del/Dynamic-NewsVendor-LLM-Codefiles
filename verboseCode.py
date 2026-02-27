import math
import random
import numpy as np
import time

# -----------------------------
#  Utility + Gumbel Generation
# -----------------------------
def draw_gumbel(mu=1.0):
    U = random.random()
    return -mu * math.log(-math.log(U))

def compute_utilities(c_utility, mu):
    eps = [draw_gumbel(mu) for _ in c_utility]
    utilities = [u + e for u, e in zip(c_utility, eps)]
    utilities = [0.0] + utilities
    return utilities

# -----------------------------
#  Dynamic Newsvendor Simulation
# -----------------------------
def simulate_dynamic_newsvendor(
    products,
    num_customers,
    c_utility,
    mu,
    init_level,
    price,
    cost
):
    inventory = np.array(init_level, dtype=int)
    price = np.array(price, dtype=float)
    cost = np.array(cost, dtype=float)
    c_utility = np.array(c_utility, dtype=float)

    total_profit = 0.0

    for t in range(num_customers):
        in_stock_indices = [j for j in products if inventory[j] > 0]
        utilities = compute_utilities(c_utility, mu)
        choice_set = [0] + [j + 1 for j in in_stock_indices]
        chosen_index = max(choice_set, key=lambda idx: utilities[idx])

        if chosen_index != 0:
            product_chosen = chosen_index - 1
            inventory[product_chosen] -= 1
            total_profit += price[product_chosen] - cost[product_chosen]

    return total_profit

# -----------------------------
#  Manual User Input + Runtime
# -----------------------------
if __name__ == "__main__":

    products = list(range(10))
    num_customers = 30
    c_utility = [6 + i for i in range(10)]
    mu = 1.0
    init_level = [3] * 10
    price = [9] * 10
    cost = [5] * 10

    start = time.perf_counter()
    profit = simulate_dynamic_newsvendor(
        products,
        num_customers,
        c_utility,
        mu,
        init_level,
        price,
        cost
    )
    end = time.perf_counter()

    print("Total Profit:", profit)
    print("Runtime (seconds):", end - start)