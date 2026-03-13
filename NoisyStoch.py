# import numpy as np
# import csv
# import time

# # -----------------------------
# # Problem parameters
# # -----------------------------
# num_products = 10
# num_customers = 10

# initial_inventory = np.array([3] * num_products)
# prices = np.arange(6, 16)  # [6,7,...,15]
# costs = np.array([1] * num_products)
# utilities = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15])
# mu = 1.5

# iterations = 10000
# output_file = "noisy_results.csv"

# # -----------------------------
# # Gumbel noise
# # -----------------------------
# def gumbel_noise(scale, size):
#     return np.random.gumbel(loc=0.0, scale=scale, size=size)

# # -----------------------------
# # One iteration of the system
# # -----------------------------
# def run_single_iteration():
#     inv = initial_inventory.copy()
#     bought = np.zeros(num_products, dtype=int)

#     for _ in range(num_customers):
#         # S(x_t): set of in-stock products
#         in_stock = np.where(inv > 0)[0]
#         if len(in_stock) == 0:
#             break

#         # Utility + Gumbel noise
#         u = utilities[in_stock] + gumbel_noise(mu, len(in_stock))

#         # Customer chooses argmax
#         choice = in_stock[np.argmax(u)]

#         # Update inventory and purchases
#         inv[choice] -= 1
#         bought[choice] += 1

#     # Profit = revenue - total initial cost
#     revenue = np.sum(bought * prices)
#     total_cost = np.sum(initial_inventory * costs)
#     profit = revenue - total_cost

#     return bought, profit

# # -----------------------------
# # Run many iterations and save CSV
# # -----------------------------
# start = time.time()

# with open(output_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     header = ["iteration"] + [f"x{i+1}" for i in range(num_products)] + ["profit"]
#     writer.writerow(header)

#     for it in range(1, iterations + 1):
#         bought, profit = run_single_iteration()
#         writer.writerow([it] + bought.tolist() + [profit])

# end = time.time()
# avg_runtime = (end - start) / iterations

# print(f"Completed {iterations} iterations.")
# print(f"Average runtime per iteration: {avg_runtime:.6f} seconds")
# print(f"Results saved to {output_file}")

import csv
import numpy as np
from time import perf_counter as shoppingtime

# ------------------------------
# Parameters (from the prompt)
# ------------------------------
num_products = 10
num_customers = 10
initial_inventory_per_product = 3
prices = np.array([6,7,8,9,10,11,12,13,14,15], dtype=float)
costs = np.ones(num_products, dtype=float) * 1.0
utility = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)
mu = 1.5  # Gumbel scale

# Constant total inventory cost (all initial inventory is paid up-front)
initial_inventory = np.ones(num_products, dtype=int) * initial_inventory_per_product
fixed_inventory_cost = float(np.sum(costs * initial_inventory))

# RNG seed for reproducibility
np.random.seed(42)

# ------------------------------
# Simulation helpers
# ------------------------------
def run_one_iteration():
    remaining = initial_inventory.copy()
    bought = np.zeros(num_products, dtype=int)

    for _ in range(num_customers):
        in_stock_idx = np.where(remaining > 0)[0]
        if in_stock_idx.size == 0:
            break
        # Gumbel noise for currently available items
        eps = np.random.gumbel(loc=0.0, scale=mu, size=in_stock_idx.size)
        # Deterministic utility + random shock on in-stock set
        total_u = utility[in_stock_idx] + eps
        choice_local = int(np.argmax(total_u))
        choice_global = int(in_stock_idx[choice_local])

        # Fulfill purchase
        bought[choice_global] += 1
        remaining[choice_global] -= 1

    # Profit = revenue from sold units minus cost of all initially stocked units
    revenue = float(np.dot(prices, bought))
    profit = revenue - fixed_inventory_cost
    return bought, profit

# ------------------------------
# Run many iterations & record
# ------------------------------
num_iterations = 10_000
out_path = 'noisy_results.csv'

start = shoppingtime()
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header: iteration,x1,...,x10,profit
    header = ['iteration'] + [f'x{i}' for i in range(1, num_products+1)] + ['profit']
    writer.writerow(header)

    for it in range(1, num_iterations+1):
        bought, profit = run_one_iteration()
        row = [it] + bought.tolist() + [profit]
        writer.writerow(row)
end = shoppingtime()

total_time = end - start
avg_time_per_iter_ms = (total_time / num_iterations) * 1000.0

# Quick sanity checks and a small preview
import pandas as pd

# Read a small sample to preview
sample_df = pd.read_csv(out_path, nrows=5)

# Aggregate check: average total units sold should be <= number of customers
full_df = pd.read_csv(out_path)
full_df['total_units_sold'] = full_df[[f'x{i}' for i in range(1, num_products+1)]].sum(axis=1)
mean_units_sold = float(full_df['total_units_sold'].mean())
min_units_sold = int(full_df['total_units_sold'].min())
max_units_sold = int(full_df['total_units_sold'].max())

print(f"Results saved to: {out_path}")
print(f"Average time per iteration: {avg_time_per_iter_ms:.4f} ms (over {num_iterations} iterations)")
print("Preview of first 5 rows:")
print(sample_df)
print(f"Sanity check — total units sold per iteration: mean={mean_units_sold:.2f}, min={min_units_sold}, max={max_units_sold}")
