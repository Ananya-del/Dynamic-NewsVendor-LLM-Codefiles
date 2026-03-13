# # # # import numpy as np
# # # # import csv
# # # # import time

# # # # # -----------------------------
# # # # # Problem Parameters
# # # # # -----------------------------
# # # # num_products = 10
# # # # num_customers = 10
# # # # initial_inventory = np.array([3] * num_products)

# # # # prices = np.arange(6, 16)          # [6,7,8,...,15]
# # # # costs = np.array([1] * num_products)
# # # # utilities = np.array([6,6,6,9,9,9,12,12,12,15])
# # # # mu = 1.5

# # # # iterations = 10000
# # # # output_file = "struc_results.csv"

# # # # # -----------------------------
# # # # # Helper: Gumbel noise
# # # # # -----------------------------
# # # # def gumbel_noise(mu_scale, size):
# # # #     return np.random.gumbel(loc=0.0, scale=mu_scale, size=size)

# # # # # -----------------------------
# # # # # Simulation for ONE iteration
# # # # # -----------------------------
# # # # def run_single_iteration():
# # # #     inv = initial_inventory.copy()
# # # #     bought = np.zeros(num_products, dtype=int)

# # # #     for _ in range(num_customers):

# # # #         # S(x_t): indices of products still in stock
# # # #         in_stock = np.where(inv > 0)[0]

# # # #         # If nothing is in stock (unlikely with these params), skip
# # # #         if len(in_stock) == 0:
# # # #             break

# # # #         # Compute utilities for in-stock products
# # # #         u = utilities[in_stock] + gumbel_noise(mu, len(in_stock))

# # # #         # Customer chooses argmax
# # # #         choice_idx = in_stock[np.argmax(u)]

# # # #         # Update inventory and purchase count
# # # #         inv[choice_idx] -= 1
# # # #         bought[choice_idx] += 1

# # # #     # Profit = revenue - total initial cost
# # # #     revenue = np.sum(bought * prices)
# # # #     total_cost = np.sum(initial_inventory * costs)
# # # #     profit = revenue - total_cost

# # # #     return bought, profit

# # # # # -----------------------------
# # # # # Run 10,000 iterations + CSV
# # # # # -----------------------------
# # # # start = time.time()

# # # # with open(output_file, "w", newline="") as f:
# # # #     writer = csv.writer(f)
# # # #     header = ["iteration"] + [f"x{i+1}" for i in range(num_products)] + ["profit"]
# # # #     writer.writerow(header)

# # # #     for it in range(1, iterations + 1):
# # # #         bought, profit = run_single_iteration()
# # # #         writer.writerow([it] + bought.tolist() + [profit])

# # # # end = time.time()
# # # # avg_runtime = (end - start) / iterations

# # # # print(f"Completed {iterations} iterations.")
# # # # print(f"Average runtime per iteration: {avg_runtime:.6f} seconds")
# # # # print(f"Results saved to {output_file}")

# # # import numpy as np
# # # import csv
# # # import time

# # # # -----------------------------
# # # # Problem Parameters
# # # # -----------------------------
# # # num_products = 10
# # # num_customers = 10

# # # initial_inventory = np.array([3] * num_products)
# # # prices = np.arange(6, 16)  # 6,7,...,15
# # # costs = np.array([1] * num_products)
# # # utilities = np.array([6,6,6,9,9,9,12,12,12,15])
# # # mu = 1.5

# # # iterations = 10000
# # # output_file = "struc_results.csv"

# # # # -----------------------------
# # # # Gumbel noise generator
# # # # -----------------------------
# # # def gumbel_noise(scale, size):
# # #     return np.random.gumbel(loc=0.0, scale=scale, size=size)

# # # # -----------------------------
# # # # Run ONE iteration
# # # # -----------------------------
# # # def run_single_iteration():
# # #     inv = initial_inventory.copy()
# # #     bought = np.zeros(num_products, dtype=int)

# # #     for _ in range(num_customers):

# # #         # S(x_t): indices of products still in stock
# # #         in_stock = np.where(inv > 0)[0]

# # #         if len(in_stock) == 0:
# # #             break

# # #         # Utility + Gumbel noise
# # #         u = utilities[in_stock] + gumbel_noise(mu, len(in_stock))

# # #         # Customer chooses argmax
# # #         choice = in_stock[np.argmax(u)]

# # #         # Update inventory and purchase count
# # #         inv[choice] -= 1
# # #         bought[choice] += 1

# # #     # Profit = revenue - total initial cost
# # #     revenue = np.sum(bought * prices)
# # #     total_cost = np.sum(initial_inventory * costs)
# # #     profit = revenue - total_cost

# # #     return bought, profit

# # # # -----------------------------
# # # # Run all iterations + write CSV
# # # # -----------------------------
# # # start = time.time()

# # # with open(output_file, "w", newline="") as f:
# # #     writer = csv.writer(f)
# # #     header = ["iteration"] + [f"x{i+1}" for i in range(num_products)] + ["profit"]
# # #     writer.writerow(header)

# # #     for it in range(1, iterations + 1):
# # #         bought, profit = run_single_iteration()
# # #         writer.writerow([it] + bought.tolist() + [profit])

# # # end = time.time()
# # # avg_runtime = (end - start) / iterations

# # # print(f"Completed {iterations} iterations.")
# # # print(f"Average runtime per iteration: {avg_runtime:.6f} seconds")
# # # print(f"Results saved to {output_file}")

# # import numpy as np
# # import csv
# # import time

# # # -----------------------------
# # # Problem Parameters
# # # -----------------------------
# # num_products = 10
# # num_customers = 10

# # initial_inventory = np.array([3] * num_products)
# # prices = np.arange(6, 16)  # 6,7,...,15
# # costs = np.array([1] * num_products)
# # utilities = np.array([6,6,6,9,9,9,12,12,12,15])
# # mu = 1.5

# # iterations = 10000
# # output_file = "struc_results.csv"

# # # -----------------------------
# # # Gumbel noise generator
# # # -----------------------------
# # def gumbel_noise(scale, size):
# #     return np.random.gumbel(loc=0.0, scale=scale, size=size)

# # # -----------------------------
# # # Run ONE iteration
# # # -----------------------------
# # def run_single_iteration():
# #     inv = initial_inventory.copy()
# #     bought = np.zeros(num_products, dtype=int)

# #     for _ in range(num_customers):

# #         # S(x_t): indices of products still in stock
# #         in_stock = np.where(inv > 0)[0]

# #         if len(in_stock) == 0:
# #             break

# #         # Utility + Gumbel noise
# #         u = utilities[in_stock] + gumbel_noise(mu, len(in_stock))

# #         # Customer chooses argmax
# #         choice = in_stock[np.argmax(u)]

# #         # Update inventory and purchase count
# #         inv[choice] -= 1
# #         bought[choice] += 1

# #     # Profit = revenue - total initial cost
# #     revenue = np.sum(bought * prices)
# #     total_cost = np.sum(initial_inventory * costs)
# #     profit = revenue - total_cost

# #     return bought, profit

# # # -----------------------------
# # # Run all iterations + write CSV
# # # -----------------------------
# # start = time.time()

# # with open(output_file, "w", newline="") as f:
# #     writer = csv.writer(f)
# #     header = ["iteration"] + [f"x{i+1}" for i in range(num_products)] + ["profit"]
# #     writer.writerow(header)

# #     for it in range(1, iterations + 1):
# #         bought, profit = run_single_iteration()
# #         writer.writerow([it] + bought.tolist() + [profit])

# # end = time.time()
# # avg_runtime = (end - start) / iterations

# # print(f"Completed {iterations} iterations.")
# # print(f"Average runtime per iteration: {avg_runtime:.6f} seconds")
# # print(f"Results saved to {output_file}")

# import numpy as np
# import csv
# import time

# # -----------------------------
# # Dynamic Newsvendor Simulation
# # -----------------------------
# # Problem setup per user spec:
# # - product int (number of products)
# # - price constant[]
# # - cost constant[]
# # - initial inventory []
# # - customers arrive sequentially
# # - Each customer always purchases exactly one unit from the in-stock set S(x_t)
# # - Choice = argmax_j { u_j + epsilon_j }, epsilon ~ Gumbel(0, mu), u_j constant
# # - Profit per iteration = sum_j price_j * x_j  - sum_j cost_j * initial_inventory_j
# # - Save results to CSV with header: 'iteration,x1,...,x10,profit'
# # - Run 10,000 iterations and measure average runtime

# # ----- Initial conditions -----
# num_products = 10
# num_customers = 10
# initial_inventory = np.full(num_products, 3, dtype=int)  # 3 for all products
# prices = np.arange(6, 16)  # [6, 7, ..., 15]
# costs = np.ones(num_products)  # 1 for all products
# u = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)  # constant utility vector
# mu = 1.5  # Gumbel scale

# # Precompute constant cost of all initial inventory (sunk cost each iteration)
# init_total_cost = float(np.dot(costs, initial_inventory))

# # Helper to run one iteration and return purchases (x) and profit
# def run_single_iteration():
#     inv = initial_inventory.copy()
#     x = np.zeros(num_products, dtype=int)

#     # For each arriving customer t: form S(x_t) = {j : inventory_j > 0}
#     for _ in range(num_customers):
#         S_xt = np.where(inv > 0)[0]  # set of in-stock products
#         # Draw Gumbel noise for products in S(x_t)
#         eps = np.random.gumbel(loc=0.0, scale=mu, size=S_xt.size)
#         # Total utility for available items: u_j + eps_j
#         tot_util = u[S_xt] + eps
#         # Choose product with maximum utility
#         chosen_local = np.argmax(tot_util)
#         chosen_prod = S_xt[chosen_local]
#         # Update inventory and purchases
#         inv[chosen_prod] -= 1
#         x[chosen_prod] += 1

#     # Profit = revenue - total cost of all initial inventory
#     revenue = float(np.dot(prices, x))
#     profit = revenue - init_total_cost
#     return x, profit

# # Simulation parameters
# n_iter = 10_000
# output_path = 'struc_results.csv'

# # Run and time
# start = time.perf_counter()
# rows = []
# for it in range(1, n_iter + 1):
#     x, profit = run_single_iteration()
#     rows.append([it, *x.tolist(), float(profit)])
# elapsed = time.perf_counter() - start
# avg_runtime_per_iter_ms = (elapsed / n_iter) * 1000

# # Write CSV with required header
# header = ['iteration'] + [f'x{i}' for i in range(1, num_products + 1)] + ['profit']
# with open(output_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerows(rows)

# # (Optional) quick summary
# print({
#     'csv_path': output_path,
#     'n_iterations': n_iter,
#     'total_runtime_seconds': elapsed,
#     'avg_runtime_per_iteration_ms': avg_runtime_per_iter_ms,
# })

import numpy as np
import csv
import time

# ------------------------------------------------------------
# Dynamic Newsvendor Problem (Sequential Choice with Gumbel)
# ------------------------------------------------------------
# 1) Inputs: product int, price[], cost[], initial_inventory[]
# 2) S(x_t): set of in-stock products at time t (inventory > 0)
# 3) Choice model: argmax_j { u_j + ε_j }, ε_j ~ Gumbel(0, μ), u_j constant
# 4) Customers arrive sequentially; each purchases at most one unit; here they always purchase
# 5) Profit per iteration = sum_j price_j * x_j  − sum_j cost_j * initial_inventory_j
# 6) Save results to CSV: 'iteration,x1,...,x10,profit'
# 7) Time the run; perform 10,000 iterations and save to CSV

# ----- Initial Conditions -----
num_products = 10
num_customers = 10
initial_inventory = np.full(num_products, 3, dtype=int)  # 3 for all products
prices = np.arange(6, 16)  # [6, 7, 8, ..., 15]
costs = np.ones(num_products, dtype=float)  # 1 for all products
u = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)  # constant utility
mu = 1.5  # Gumbel scale

# Precompute total cost of all initial inventory (charged regardless of sales)
init_total_cost = float(np.dot(costs, initial_inventory))

def run_single_iteration():
    """Run one iteration: simulate 10 sequential customers.
    Returns purchase vector x (length 10) and profit.
    """
    inv = initial_inventory.copy()
    x = np.zeros(num_products, dtype=int)

    for _ in range(num_customers):
        # S(x_t) = indices of products with inventory remaining
        S_xt = np.where(inv > 0)[0]

        # Draw Gumbel noise for the currently available alternatives
        eps = np.random.gumbel(loc=0.0, scale=mu, size=S_xt.size)

        # Total utility for available products
        tot_util = u[S_xt] + eps

        # Choose product with maximum utility
        chosen_local = int(np.argmax(tot_util))
        chosen_prod = int(S_xt[chosen_local])

        # Update inventory and purchases
        inv[chosen_prod] -= 1
        x[chosen_prod] += 1

    # Profit: revenue minus total cost of all initial inventory (sunk this period)
    revenue = float(np.dot(prices, x))
    profit = revenue - init_total_cost
    return x, profit

# Simulation parameters
n_iter = 10_000
output_csv = 'struc_results.csv'

# Run simulation and time it
start = time.perf_counter()
rows = []
for it in range(1, n_iter + 1):
    x, profit = run_single_iteration()
    rows.append([it, *x.tolist(), float(profit)])
elapsed = time.perf_counter() - start
avg_runtime_per_iter_ms = (elapsed / n_iter) * 1000.0

# Write CSV
header = ['iteration'] + [f'x{i}' for i in range(1, num_products + 1)] + ['profit']
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

# Quick summary for the user
profits = np.array([r[-1] for r in rows], dtype=float)
summary = {
    'csv_path': output_csv,
    'iterations': n_iter,
    'total_runtime_seconds': elapsed,
    'avg_runtime_per_iteration_ms': avg_runtime_per_iter_ms,
    'min_profit': float(profits.min()),
    'max_profit': float(profits.max()),
    'mean_profit': float(profits.mean()),
}
print(summary)