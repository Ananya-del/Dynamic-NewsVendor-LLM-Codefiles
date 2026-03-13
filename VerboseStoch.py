# # # # import csv, time
# # # # import numpy as np

# # # # # ------------------------
# # # # # Dynamic Newsvendor Simulation
# # # # # ------------------------
# # # # # Settings from the user
# # # # num_products = 10
# # # # num_customers = 10
# # # # initial_inventory_per_product = 3
# # # # prices = np.array([6,7,8,9,10,11,12,13,14,15], dtype=float)
# # # # costs = np.ones(num_products, dtype=float)
# # # # utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)
# # # # mu = 1.5  # Gumbel scale

# # # # iterations = 10_000
# # # # output_path = "verb_results.csv"

# # # # # Optional: set a seed for reproducibility (comment out if true randomness preferred)
# # # # np.random.seed(42)

# # # # header = ["iteration"] + [f"x{i}" for i in range(1, num_products+1)] + ["profit"]
# # # # rows = []

# # # # start = time.time()
# # # # for it in range(1, iterations+1):
# # # #     inventory = np.full(num_products, initial_inventory_per_product, dtype=int)
# # # #     purchases = np.zeros(num_products, dtype=int)

# # # #     for c in range(num_customers):
# # # #         available = np.where(inventory > 0)[0]
# # # #         # Draw one Gumbel shock per available product
# # # #         eps = np.random.gumbel(loc=0.0, scale=mu, size=available.shape[0])
# # # #         util_draw = utilities[available] + eps
# # # #         # Choose product with max realized utility
# # # #         chosen_idx = available[np.argmax(util_draw)]
# # # #         # Update inventory and purchases
# # # #         inventory[chosen_idx] -= 1
# # # #         purchases[chosen_idx] += 1

# # # #     profit = float(np.sum((prices - costs) * purchases))
# # # #     rows.append([it] + purchases.tolist() + [profit])

# # # # end = time.time()

# # # # total_runtime = end - start
# # # # avg_runtime = total_runtime / iterations

# # # # # Write CSV
# # # # with open(output_path, mode="w", newline="") as f:
# # # #     writer = csv.writer(f)
# # # #     writer.writerow(header)
# # # #     writer.writerows(rows)

# # # # # Provide a couple of quick sanity summaries
# # # # aggregate_purchases = np.sum([r[1:1+num_products] for r in rows], axis=0)
# # # # aggregate_profit = sum(r[-1] for r in rows)

# # # # print({
# # # #     "csv_path": output_path,
# # # #     "iterations": iterations,
# # # #     "avg_runtime_sec_per_iteration": avg_runtime,
# # # #     "total_runtime_sec": total_runtime,
# # # #     "aggregate_purchases_over_all_iterations": aggregate_purchases.tolist(),
# # # #     "aggregate_profit_over_all_iterations": aggregate_profit
# # # # })

# # # import numpy as np
# # # import csv
# # # import time

# # # # ---------------------------------------------------------
# # # # Problem Parameters
# # # # ---------------------------------------------------------
# # # n_products = 10
# # # n_customers = 10
# # # n_iterations = 10000

# # # initial_inventory = np.full(n_products, 3, dtype=int)

# # # prices = np.arange(6, 16, dtype=float)     # [6,7,...,15]
# # # costs = np.ones(n_products, dtype=float)   # all 1
# # # utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)

# # # mu = 1.5  # Gumbel scale parameter

# # # rng = np.random.default_rng(2026)  # reproducible seed


# # # # ---------------------------------------------------------
# # # # One iteration of the dynamic newsvendor simulation
# # # # ---------------------------------------------------------
# # # def run_single_iteration():
# # #     inventory = initial_inventory.copy()
# # #     bought_counts = np.zeros(n_products, dtype=int)

# # #     for _ in range(n_customers):

# # #         # Set of in-stock products S(x_t)
# # #         in_stock = np.where(inventory > 0)[0]

# # #         # If nothing in stock → no more purchases possible
# # #         if in_stock.size == 0:
# # #             break

# # #         # Deterministic utilities for in-stock products
# # #         u = utilities[in_stock]

# # #         # Gumbel noise with scale mu
# # #         eps = rng.gumbel(loc=0.0, scale=mu, size=in_stock.size)

# # #         # Total utilities
# # #         total_util = u + eps

# # #         # Choose product with maximum utility
# # #         best_idx = np.argmax(total_util)
# # #         chosen_global_idx = in_stock[best_idx]

# # #         # Update inventory and purchase count
# # #         inventory[chosen_global_idx] -= 1
# # #         bought_counts[chosen_global_idx] += 1

# # #     # Profit = total revenue - total cost of initial inventory
# # #     revenue = np.sum(bought_counts * prices)
# # #     total_cost = np.sum(initial_inventory * costs)
# # #     profit = revenue - total_cost

# # #     return bought_counts, profit


# # # # ---------------------------------------------------------
# # # # Main simulation loop
# # # # ---------------------------------------------------------
# # # def main():
# # #     output_file = "verb_results.csv"
# # #     header = ['iteration'] + [f'x{i+1}' for i in range(n_products)] + ['profit']

# # #     start = time.time()

# # #     with open(output_file, mode='w', newline='') as f:
# # #         writer = csv.writer(f)
# # #         writer.writerow(header)

# # #         for it in range(1, n_iterations + 1):
# # #             bought_counts, profit = run_single_iteration()
# # #             row = [it] + bought_counts.tolist() + [profit]
# # #             writer.writerow(row)

# # #     end = time.time()
# # #     total_runtime = end - start
# # #     avg_runtime = total_runtime / n_iterations

# # #     print(f"Completed {n_iterations} iterations")
# # #     print(f"Total runtime: {total_runtime:.4f} seconds")
# # #     print(f"Average runtime per iteration: {avg_runtime:.8f} seconds")
# # #     print(f"Saved to: {output_file}")


# # # if __name__ == "__main__":
# # #     main()

# # import csv, time
# # import numpy as np

# # # ------------------------
# # # Dynamic Newsvendor Simulation (version with profit = revenue - total initial inventory cost)
# # # ------------------------
# # # Settings from the user
# # num_products = 10
# # num_customers = 10
# # initial_inventory_per_product = 3
# # prices = np.array([6,7,8,9,10,11,12,13,14,15], dtype=float)
# # costs = np.ones(num_products, dtype=float)
# # utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)
# # mu = 1.5  # Gumbel scale

# # iterations = 10_000
# # output_path = "verb_results.csv"

# # # Optional: set a seed for reproducibility (comment out if true randomness preferred)
# # np.random.seed(42)

# # header = ["iteration"] + [f"x{i}" for i in range(1, num_products+1)] + ["profit"]
# # rows = []

# # # Precompute total cost of all initial inventory (as requested)
# # # total_initial_cost = sum_i (cost_i * initial_inventory_i)
# # # with initial_inventory_i = initial_inventory_per_product for all i
# # total_initial_cost = float(initial_inventory_per_product * np.sum(costs))

# # start = time.time()
# # for it in range(1, iterations+1):
# #     inventory = np.full(num_products, initial_inventory_per_product, dtype=int)
# #     purchases = np.zeros(num_products, dtype=int)

# #     for c in range(num_customers):
# #         available = np.where(inventory > 0)[0]
# #         # Draw one Gumbel shock per available product
# #         eps = np.random.gumbel(loc=0.0, scale=mu, size=available.shape[0])
# #         util_draw = utilities[available] + eps
# #         # Choose product with max realized utility
# #         chosen_idx = available[np.argmax(util_draw)]
# #         # Update inventory and purchases
# #         inventory[chosen_idx] -= 1
# #         purchases[chosen_idx] += 1

# #     # Profit per iteration = revenue - (total cost of all initial inventory)
# #     revenue = float(np.sum(prices * purchases))
# #     profit = revenue - total_initial_cost
# #     rows.append([it] + purchases.tolist() + [profit])

# # end = time.time()

# # total_runtime = end - start
# # avg_runtime = total_runtime / iterations

# # # Write CSV
# # with open(output_path, mode="w", newline="") as f:
# #     writer = csv.writer(f)
# #     writer.writerow(header)
# #     writer.writerows(rows)

# # # Provide quick sanity summaries
# # aggregate_purchases = np.sum([r[1:1+num_products] for r in rows], axis=0)
# # aggregate_profit = sum(r[-1] for r in rows)

# # print({
# #     "csv_path": output_path,
# #     "iterations": iterations,
# #     "avg_runtime_sec_per_iteration": avg_runtime,
# #     "total_runtime_sec": total_runtime,
# #     "aggregate_purchases_over_all_iterations": aggregate_purchases.tolist(),
# #     "aggregate_profit_over_all_iterations": aggregate_profit,
# #     "total_initial_cost_per_iteration": total_initial_cost
# # })

# import numpy as np
# import pandas as pd
# import time

# # --- Parameters ---
# N_PRODUCTS = 10
# N_CUSTOMERS = 10
# INITIAL_INVENTORY_PER_PRODUCT = 3
# PRICE = np.arange(6, 16)  # [6, 7, ..., 15]
# COST = np.ones(N_PRODUCTS)  # cost = 1 for all products
# UTILITY = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15])
# MU = 1.5  # scale parameter for Gumbel noise
# N_ITERS = 10_000

# # Precompute total initial cost (sunk cost of holding starting inventory)
# TOTAL_INITIAL_COST = float(np.sum(COST * INITIAL_INVENTORY_PER_PRODUCT))  # 10 * 1 * 3 = 30

# columns = ["iteration"] + [f"x{i}" for i in range(1, N_PRODUCTS + 1)] + ["profit"]
# results = []

# start = time.perf_counter()
# for it in range(1, N_ITERS + 1):
#     # Reset state each iteration
#     inventory = np.full(N_PRODUCTS, INITIAL_INVENTORY_PER_PRODUCT, dtype=int)
#     x = np.zeros(N_PRODUCTS, dtype=int)  # quantities purchased per product in this iteration

#     # Sequential customer arrivals; each buys exactly one unit of the best available option
#     for _ in range(N_CUSTOMERS):
#         in_stock_idx = np.where(inventory > 0)[0]
#         # Compute random utility for in-stock items only
#         eps = np.random.gumbel(loc=0.0, scale=MU, size=in_stock_idx.size)
#         utilities = UTILITY[in_stock_idx] + eps
#         # Argmax over available items
#         choice = in_stock_idx[np.argmax(utilities)]
#         # Update inventory and purchased counts
#         x[choice] += 1
#         inventory[choice] -= 1

#     # Profit: total price * amount bought minus the total cost of all initial inventory
#     profit = float(PRICE @ x - TOTAL_INITIAL_COST)

#     # Append row: iteration, x1..x10, profit
#     results.append([it] + list(map(int, x)) + [profit])

# elapsed = time.perf_counter() - start
# avg_runtime = elapsed / N_ITERS

# # Save to CSV
# out_path = "verb_results.csv"
# df = pd.DataFrame(results, columns=columns)
# df.to_csv(out_path, index=False)

# print(f"Saved results to {out_path}")
# print(f"Total iterations: {N_ITERS}")
# print(f"Total time (s): {elapsed:.6f}")
# print(f"Average runtime per iteration (s): {avg_runtime:.9f}")

# # Print a quick preview of the first 3 rows
# print("\nPreview:")
# print(df.head(3).to_string(index=False))

import numpy as np
import csv
import time

# ---------------------------------------------------------
# Problem Parameters
# ---------------------------------------------------------
n_products = 10
n_customers = 10
n_iterations = 10000

initial_inventory = np.full(n_products, 3, dtype=int)

prices = np.arange(6, 16, dtype=float)     # [6,7,...,15]
costs = np.ones(n_products, dtype=float)   # all 1
utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)

mu = 1.5  # Gumbel scale parameter

rng = np.random.default_rng(2026)  # reproducible seed


# ---------------------------------------------------------
# One iteration of the dynamic newsvendor simulation
# ---------------------------------------------------------
def run_single_iteration():
    inventory = initial_inventory.copy()
    bought_counts = np.zeros(n_products, dtype=int)

    for _ in range(n_customers):

        # Set of in-stock products S(x_t)
        in_stock = np.where(inventory > 0)[0]

        # If nothing in stock → no more purchases possible
        if in_stock.size == 0:
            break

        # Deterministic utilities for in-stock products
        u = utilities[in_stock]

        # Gumbel noise with scale mu
        eps = rng.gumbel(loc=0.0, scale=mu, size=in_stock.size)

        # Total utilities
        total_util = u + eps

        # Choose product with maximum utility
        best_idx = np.argmax(total_util)
        chosen_global_idx = in_stock[best_idx]

        # Update inventory and purchase count
        inventory[chosen_global_idx] -= 1
        bought_counts[chosen_global_idx] += 1

    # Profit = total revenue - total cost of initial inventory
    revenue = np.sum(bought_counts * prices)
    total_cost = np.sum(initial_inventory * costs)
    profit = revenue - total_cost

    return bought_counts, profit


# ---------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------
def main():
    output_file = "verb_results.csv"
    header = ['iteration'] + [f'x{i+1}' for i in range(n_products)] + ['profit']

    start = time.time()

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for it in range(1, n_iterations + 1):
            bought_counts, profit = run_single_iteration()
            row = [it] + bought_counts.tolist() + [profit]
            writer.writerow(row)

    end = time.time()
    total_runtime = end - start
    avg_runtime = total_runtime / n_iterations

    print(f"Completed {n_iterations} iterations")
    print(f"Total runtime: {total_runtime:.4f} seconds")
    print(f"Average runtime per iteration: {avg_runtime:.8f} seconds")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()