# # # # import numpy as np
# # # # import csv
# # # # import time

# # # # # ---------------------------------------------------------
# # # # # Problem Parameters
# # # # # ---------------------------------------------------------
# # # # n_products = 10
# # # # n_customers = 10
# # # # n_iterations = 10000

# # # # initial_inventory = np.full(n_products, 3, dtype=int)

# # # # prices = np.arange(6, 16, dtype=float)     # [6,7,...,15]
# # # # costs = np.ones(n_products, dtype=float)   # all 1
# # # # utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)

# # # # mu = 1.5  # Gumbel scale parameter

# # # # rng = np.random.default_rng(2026)  # reproducible seed


# # # # # ---------------------------------------------------------
# # # # # One iteration of the dynamic newsvendor simulation
# # # # # ---------------------------------------------------------
# # # # def run_single_iteration():
# # # #     inventory = initial_inventory.copy()
# # # #     bought_counts = np.zeros(n_products, dtype=int)

# # # #     for _ in range(n_customers):

# # # #         # Set of in-stock products S(x_t)
# # # #         in_stock = np.where(inventory > 0)[0]

# # # #         # If nothing in stock → no more purchases possible
# # # #         if in_stock.size == 0:
# # # #             break

# # # #         # Deterministic utilities for in-stock products
# # # #         u = utilities[in_stock]

# # # #         # Gumbel noise with scale mu
# # # #         eps = rng.gumbel(loc=0.0, scale=mu, size=in_stock.size)

# # # #         # Total utilities
# # # #         total_util = u + eps

# # # #         # Choose product with maximum utility
# # # #         best_idx = np.argmax(total_util)
# # # #         chosen_global_idx = in_stock[best_idx]

# # # #         # Update inventory and purchase count
# # # #         inventory[chosen_global_idx] -= 1
# # # #         bought_counts[chosen_global_idx] += 1

# # # #     # Profit = total revenue - total cost of initial inventory
# # # #     revenue = np.sum(bought_counts * prices)
# # # #     total_cost = np.sum(initial_inventory * costs)
# # # #     profit = revenue - total_cost

# # # #     return bought_counts, profit


# # # # # ---------------------------------------------------------
# # # # # Main simulation loop
# # # # # ---------------------------------------------------------
# # # # def main():
# # # #     output_file = "amb_results.csv"
# # # #     header = ['iteration'] + [f'x{i+1}' for i in range(n_products)] + ['profit']

# # # #     start = time.time()

# # # #     with open(output_file, mode='w', newline='') as f:
# # # #         writer = csv.writer(f)
# # # #         writer.writerow(header)

# # # #         for it in range(1, n_iterations + 1):
# # # #             bought_counts, profit = run_single_iteration()
# # # #             row = [it] + bought_counts.tolist() + [profit]
# # # #             writer.writerow(row)

# # # #     end = time.time()
# # # #     total_runtime = end - start
# # # #     avg_runtime = total_runtime / n_iterations

# # # #     print(f"Completed {n_iterations} iterations")
# # # #     print(f"Total runtime: {total_runtime:.4f} seconds")
# # # #     print(f"Average runtime per iteration: {avg_runtime:.8f} seconds")
# # # #     print(f"Saved to: {output_file}")


# # # # if __name__ == "__main__":
# # # #     main()

# # # import numpy as np
# # # import time
# # # import csv
# # # import os

# # # # -----------------------------
# # # # Dynamic Newsvendor Simulation
# # # # -----------------------------
# # # # Initial conditions (from user request)
# # # N_PRODUCTS = 10
# # # N_CUSTOMERS = 10
# # # INITIAL_INVENTORY_PER_PRODUCT = 3
# # # PRICES = np.arange(6, 16, dtype=int)                 # [6, 7, ..., 15]
# # # COSTS = np.ones(N_PRODUCTS, dtype=int)               # all ones
# # # UTILITIES = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15], dtype=float)
# # # MU = 1.5                                             # Gumbel scale
# # # N_ITER = 10_000

# # # # (Optional) set a fixed seed for reproducibility. Comment out for fully random runs.
# # # np.random.seed(42)

# # # # Precompute constant vectors
# # # INITIAL_INVENTORY = np.full(N_PRODUCTS, INITIAL_INVENTORY_PER_PRODUCT, dtype=int)
# # # PROCUREMENT_COST_TOTAL = int(np.sum(INITIAL_INVENTORY * COSTS))  # cost paid on initial inventory

# # # # Output file
# # # OUTFILE = "amb_results.csv"

# # # def simulate_one_iteration():
# # #     """Simulate one iteration of 10 customers arriving sequentially.
# # #     Each customer chooses the in-stock product that maximizes (utility + Gumbel(0, MU)).
# # #     Returns: (purchase_counts[10], profit)
# # #     """
# # #     inventory = INITIAL_INVENTORY.copy()
# # #     purchases = np.zeros(N_PRODUCTS, dtype=int)

# # #     for _ in range(N_CUSTOMERS):
# # #         available_mask = inventory > 0
# # #         # Draw Gumbel noise for each product, mask out-of-stock items to -inf utility
# # #         noise = np.random.gumbel(loc=0.0, scale=MU, size=N_PRODUCTS)
# # #         total_util = np.where(available_mask, UTILITIES + noise, -np.inf)
# # #         chosen = int(np.argmax(total_util))
# # #         # Update inventory and purchase counts
# # #         inventory[chosen] -= 1
# # #         purchases[chosen] += 1

# # #     # Profit: revenue from purchases minus procurement cost of initial inventory
# # #     revenue = int(purchases @ PRICES)
# # #     profit = int(revenue - PROCUREMENT_COST_TOTAL)
# # #     return purchases, profit

# # # # Run simulation and time it
# # # start = time.time()

# # # rows = []
# # # for it in range(1, N_ITER + 1):
# # #     counts, profit = simulate_one_iteration()
# # #     rows.append([it, *counts.tolist(), profit])

# # # end = time.time()

# # # total_time_sec = end - start
# # # avg_time_per_iter_ms = (total_time_sec / N_ITER) * 1000.0

# # # # Write CSV with the requested header
# # # header = [
# # #     "iteration",
# # #     "x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
# # #     "profit"
# # # ]

# # # with open(OUTFILE, mode="w", newline="") as f:
# # #     writer = csv.writer(f)
# # #     writer.writerow(header)
# # #     writer.writerows(rows)

# # # abs_path = os.path.abspath(OUTFILE)
# # # print(f"Saved results to: {abs_path}")
# # # print(f"Total runtime (s): {total_time_sec:.6f}")
# # # print(f"Average runtime per iteration (ms): {avg_time_per_iter_ms:.6f}")

# # import numpy as np
# # import pandas as pd
# # import time

# # # ------------------------------
# # # Dynamic Newsvendor Simulation
# # # ------------------------------
# # # Assumptions implemented per request:
# # # - 10 products, 10 customers per iteration
# # # - Initial inventory: 3 units for each product (cost incurred on initial inventory)
# # # - Each customer always purchases exactly one available product
# # # - Choice rule: argmax_j { Utility_j + Gumbel_j }, restricted to in-stock products
# # # - Gumbel noise ~ Gumbel(loc=0, scale=Mu), with Mu = 1.5
# # # - Profit per iteration = sum_sold(price_j) - cost_per_unit * sum(initial_inventory)
# # # - Save 10,000 iterations to CSV with header:
# # #   'iteration,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,profit'

# # # Parameters
# # N_PRODUCTS = 10
# # N_CUSTOMERS = 10
# # initial_inventory = np.full(N_PRODUCTS, 3, dtype=int)
# # prices = np.arange(6, 16)  # [6,7,...,15]
# # cost_per_unit = 1
# # utility = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15], dtype=float)
# # Mu = 1.5
# # N_ITER = 10000

# # # Storage for results
# # X = np.zeros((N_ITER, N_PRODUCTS), dtype=int)
# # profits = np.zeros(N_ITER, dtype=float)

# # # Timing
# # start = time.perf_counter()

# # # Simulation
# # for it in range(N_ITER):
# #     inv = initial_inventory.copy()
# #     sold = np.zeros(N_PRODUCTS, dtype=int)
# #     revenue = 0.0

# #     for _ in range(N_CUSTOMERS):
# #         avail = np.where(inv > 0)[0]
# #         if avail.size == 0:
# #             # No inventory left (not expected with given params), break safely
# #             break
# #         # Draw Gumbel noise for available products and pick argmax
# #         noise = np.random.gumbel(loc=0.0, scale=Mu, size=avail.size)
# #         total_util = utility[avail] + noise
# #         chosen_local_idx = np.argmax(total_util)
# #         chosen = avail[chosen_local_idx]

# #         # Update inventory, sales, and revenue
# #         inv[chosen] -= 1
# #         sold[chosen] += 1
# #         revenue += prices[chosen]

# #     # Profit: revenue minus cost of initial inventory (newsvendor-style purchase up front)
# #     total_cost = cost_per_unit * initial_inventory.sum()
# #     profits[it] = revenue - total_cost
# #     X[it, :] = sold

# # end = time.perf_counter()

# # total_runtime_sec = end - start
# # avg_runtime_per_iter_ms = (total_runtime_sec / N_ITER) * 1000.0

# # # Prepare and save CSV
# # cols = ["iteration"] + [f"x{i}" for i in range(1, N_PRODUCTS + 1)] + ["profit"]
# # df = pd.DataFrame(
# #     np.column_stack([
# #         np.arange(1, N_ITER + 1),
# #         X,
# #         profits
# #     ]),
# #     columns=cols
# # )

# # # Ensure integer types for iteration and xi columns
# # for c in ["iteration"] + [f"x{i}" for i in range(1, N_PRODUCTS + 1)]:
# #     df[c] = df[c].astype(int)

# # # Save
# # output_path = "amb_results.csv"
# # df.to_csv(output_path, index=False)

# # # Optional: quick printout
# # print({
# #     "output_file": output_path,
# #     "rows": len(df),
# #     "columns": list(df.columns),
# #     "total_runtime_sec": round(total_runtime_sec, 4),
# #     "avg_runtime_per_iteration_ms": round(avg_runtime_per_iter_ms, 6),
# #     "profit_mean": float(df["profit"].mean()),
# #     "profit_std": float(df["profit"].std(ddof=1)),
# # })
# # print(df.head(5))

# import numpy as np
# import pandas as pd
# import time

# # ------------------------------
# # Parameters (from your specs)
# # ------------------------------
# N_PRODUCTS = 10
# N_CUSTOMERS = 10
# INIT_INVENTORY = 3
# PRICES = np.arange(6, 16)  # [6,7,8,...,15]
# COST = 1
# UTILS = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15], dtype=float)
# MU = 1.5  # Gumbel scale parameter
# N_ITER = 10_000

# # Optional: set seed for reproducibility
# np.random.seed(42)

# # Storage for results: iteration, x1..x10, profit
# cols = ["iteration"] + [f"x{i}" for i in range(1, N_PRODUCTS+1)] + ["profit"]
# results = np.zeros((N_ITER, len(cols)), dtype=float)

# start = time.time()
# for it in range(N_ITER):
#     inventory = np.full(N_PRODUCTS, INIT_INVENTORY, dtype=int)
#     qty = np.zeros(N_PRODUCTS, dtype=int)

#     for _ in range(N_CUSTOMERS):
#         available_idx = np.flatnonzero(inventory > 0)
#         # With 30 total units and 10 customers, availability should never be empty,
#         # but we guard anyway.
#         if available_idx.size == 0:
#             break

#         # Additive random utility with Gumbel noise (scale=MU)
#         noise = np.random.gumbel(loc=0.0, scale=MU, size=available_idx.size)
#         scores = UTILS[available_idx] + noise
#         chosen_local = np.argmax(scores)
#         chosen = available_idx[chosen_local]

#         qty[chosen] += 1
#         inventory[chosen] -= 1

#     profit = np.sum(qty * (PRICES - COST))

#     # Save row
#     results[it, 0] = it + 1  # iteration index starting at 1
#     results[it, 1:1+N_PRODUCTS] = qty
#     results[it, -1] = profit

# end = time.time()

# total_runtime_sec = end - start
# avg_runtime_per_iteration_ms = (total_runtime_sec / N_ITER) * 1000.0

# # Save to CSV
# df = pd.DataFrame(results, columns=cols)
# # Cast integer-looking columns
# int_cols = [c for c in cols if c.startswith('x')] + ["iteration", "profit"]
# for c in int_cols:
#     df[c] = df[c].astype(int)

# output_path = "amb_results.csv"
# df.to_csv(output_path, index=False)

# print(f"Average runtime per iteration (ms): {avg_runtime_per_iteration_ms:.4f}")
# print(f"Total runtime (s): {total_runtime_sec:.3f}")
# print(f"Saved to: {output_path}")

import numpy as np
import time

# ----- Initial conditions -----
num_products = 10
num_customers = 10
initial_inventory = np.full(num_products, 3, dtype=int)
prices = np.arange(6, 16)  # [6, 7, ..., 15]
costs = np.ones(num_products)
base_utility = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)
mu = 1.5  # Gumbel scale parameter

# Assumption: Deterministic utility = base_utility - price
# Total utility for in-stock items = (base_utility - price) + Gumbel(0, mu)

def run_single_iteration():
    inv = initial_inventory.copy()
    profit = 0.0
    det_util = base_utility - prices  # deterministic component

    for _ in range(num_customers):
        # Indices of in-stock products
        in_stock = np.where(inv > 0)[0]

        # Draw Gumbel noise for currently available products
        noise = np.random.gumbel(loc=0.0, scale=mu, size=in_stock.size)

        # Compute total utilities for available set
        tot_util = det_util[in_stock] + noise

        # Choose product with maximum utility
        chosen_idx_local = np.argmax(tot_util)
        chosen_prod = in_stock[chosen_idx_local]

        # Update inventory and profit
        inv[chosen_prod] -= 1
        profit += (prices[chosen_prod] - costs[chosen_prod])

    return profit

# Run many iterations and measure runtime
n_iter = 10_000
profits = np.empty(n_iter)
start = time.perf_counter()
for i in range(n_iter):
    profits[i] = run_single_iteration()
elapsed = time.perf_counter() - start

avg_runtime_per_iter_ms = (elapsed / n_iter) * 1000

# Quantiles of the profit distribution
quantiles = {q: float(np.percentile(profits, q)) for q in [5, 10, 50, 90, 95]}

print({
    'n_iterations': n_iter,
    'avg_runtime_per_iteration_ms': avg_runtime_per_iter_ms,
    'total_runtime_seconds': elapsed,
    'profit_quantiles': quantiles,
    'mean_profit': float(np.mean(profits)),
    'std_profit': float(np.std(profits, ddof=1)),
})