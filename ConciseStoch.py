# ## This file has all concise codes
# ## Starting with C1

# import numpy as np
# import csv
# import time

# # -----------------------------
# # Problem parameters
# # -----------------------------
# n_products = 10
# n_customers = 10
# n_iterations = 1000  # change as needed

# # Initial inventory for each product
# initial_inventory = np.full(n_products, 3, dtype=int)

# # Prices 6,7,...,15
# prices = np.arange(6, 16, dtype=float)

# # Costs all 1
# costs = np.ones(n_products, dtype=float)

# # Utilities
# utilities = np.array([6, 6, 6, 9, 9, 9, 12, 12, 12, 15], dtype=float)

# # Random seed for reproducibility
# rng = np.random.default_rng(12345)

# # -----------------------------
# # Optional: Gumbel CDF function
# # -----------------------------
# def gumbel_cdf(x, mu=0.0, beta=1.0):
#     z = (x - mu) / beta
#     return np.exp(-np.exp(-z))

# # -----------------------------
# # Single iteration simulation
# # -----------------------------
# def run_single_iteration():
#     inventory = initial_inventory.copy()
#     profit = 0.0

#     # Customers arrive sequentially
#     for _ in range(n_customers):
#         # Set of in-stock products S(x_t)
#         in_stock_idx = np.where(inventory > 0)[0]

#         # If nothing in stock, only outside option is available
#         if in_stock_idx.size == 0:
#             # Outside option only → no sale
#             continue

#         # Utilities for in-stock products
#         u_instock = utilities[in_stock_idx]

#         # Draw Gumbel noise for in-stock products and outside option
#         eps_products = rng.gumbel(loc=0.0, scale=1.0, size=in_stock_idx.size)
#         eps_outside = rng.gumbel(loc=0.0, scale=1.0)

#         # Total utilities
#         total_util_products = u_instock + eps_products
#         total_util_outside = 0.0 + eps_outside  # outside option utility

#         # Choose argmax over S(x_t) ∪ {0}
#         best_product_idx = np.argmax(total_util_products)
#         best_product_utility = total_util_products[best_product_idx]

#         if best_product_utility > total_util_outside:
#             # Purchase occurs for that product
#             chosen_global_idx = in_stock_idx[best_product_idx]
#             inventory[chosen_global_idx] -= 1
#             profit += prices[chosen_global_idx] - costs[chosen_global_idx]
#         else:
#             # Outside option chosen → no purchase
#             continue

#     return inventory, profit

# # -----------------------------
# # Main simulation loop
# # -----------------------------
# def main():
#     output_file = "concise_results.csv"
#     header = ['iteration'] + [f'x{i+1}' for i in range(n_products)] + ['profit']

#     start_time = time.time()

#     with open(output_file, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

#         for it in range(1, n_iterations + 1):
#             final_inventory, profit = run_single_iteration()
#             row = [it] + final_inventory.tolist() + [profit]
#             writer.writerow(row)

#     end_time = time.time()
#     runtime = end_time - start_time
#     print(f"Simulation completed for {n_iterations} iterations.")
#     print(f"Runtime: {runtime:.4f} seconds")
#     print(f"Results saved to: {output_file}")

# if __name__ == "__main__":
#     main()

# import numpy as np
# import csv
# import time
# from typing import List, Tuple

# # -----------------------------
# # Dynamic Newsvendor Simulation
# # -----------------------------
# # Assumptions & notes:
# # 1) Customers arrive sequentially and choose the item (or outside option 0) with the highest utility.
# # 2) Utilities are u_i + epsilon_i, where epsilon_i ~ Gumbel(loc=0, scale=1), IID across alternatives.
# # 3) The outside option (0) has deterministic utility 0 and its own Gumbel noise.
# # 4) Profit per iteration = total revenue from sold units - total procurement cost of initial inventory.
# #    (No salvage value and no holding cost beyond initial procurement.)
# # 5) We record the END-OF-HORIZON inventory x1..x10 and the realized profit per iteration to CSV.

# def gumbel_cdf(x: np.ndarray, mu: float = 0.0, beta: float = 1.0) -> np.ndarray:
#     """Gumbel CDF: F(x) = exp(-exp(-(x-mu)/beta)). Provided for reference.
#     Not required for the simulation because we sample with numpy.random.gumbel.
#     """
#     z = (x - mu) / beta
#     return np.exp(-np.exp(-z))

# def simulate_one_run(
#     prices: np.ndarray,
#     costs: np.ndarray,
#     base_utils: np.ndarray,
#     init_inventory: np.ndarray,
#     n_customers: int,
#     rng: np.random.Generator,
# ) -> Tuple[np.ndarray, float]:
#     """Run a single iteration (scenario) of the dynamic newsvendor process.

#     Returns:
#         end_inventory: array of length N_products with remaining inventory.
#         profit: realized profit for this iteration.
#     """
#     n_products = len(prices)
#     # Copy inventory; pay procurement up-front (newsvendor style)
#     inventory = init_inventory.astype(int).copy()
#     procurement_cost = float(costs @ init_inventory)

#     revenue = 0.0

#     for _ in range(n_customers):
#         # Set of in-stock products S(x_t)
#         in_stock_idx = np.where(inventory > 0)[0]

#         # Build candidate set = outside option + in-stock products
#         # Outside option is coded as index -1
#         candidates = [-1]
#         if in_stock_idx.size > 0:
#             candidates.extend(in_stock_idx.tolist())

#         # Deterministic utilities for candidates
#         det_utils = []
#         for c in candidates:
#             if c == -1:
#                 det_utils.append(0.0)  # outside option base utility
#             else:
#                 det_utils.append(float(base_utils[c]))
#         det_utils = np.array(det_utils)

#         # Add Gumbel noise to each candidate
#         eps = rng.gumbel(loc=0.0, scale=1.0, size=len(candidates))
#         total_utils = det_utils + eps

#         # Choose argmax
#         choice_idx = int(np.argmax(total_utils))
#         choice = candidates[choice_idx]

#         # Fulfill sale if a product was chosen
#         if choice != -1:
#             revenue += float(prices[choice])
#             inventory[choice] -= 1
#             # (one unit per customer)

#     profit = revenue - procurement_cost
#     return inventory, float(profit)

# def run_experiment(
#     n_products: int = 10,
#     n_customers: int = 10,
#     init_units: int = 3,
#     prices_list: List[float] | None = None,
#     cost_value: float = 1.0,
#     utility_list: List[float] | None = None,
#     n_iterations: int = 1000,
#     seed: int = 42,
#     out_csv: str = "dynamic_newsvendor_results.csv",
# ) -> dict:
#     """Run many iterations and write results to CSV with header:
#     iteration,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,profit

#     Returns a dictionary with basic summary stats and runtime seconds.
#     """
#     # Defaults from your spec
#     if prices_list is None:
#         prices_list = list(range(6, 16))  # [6,7,8,...,15]
#     if utility_list is None:
#         utility_list = [6,6,6,9,9,9,12,12,12,15]

#     # Convert to arrays
#     prices = np.array(prices_list, dtype=float)
#     costs = np.full(n_products, float(cost_value))
#     base_utils = np.array(utility_list, dtype=float)
#     init_inventory = np.full(n_products, int(init_units))

#     assert len(prices) == n_products
#     assert len(base_utils) == n_products

#     rng = np.random.default_rng(seed)

#     start = time.perf_counter()

#     # Prepare CSV
#     header = ["iteration"] + [f"x{i}" for i in range(1, n_products+1)] + ["profit"]
#     rows_written = 0

#     with open(out_csv, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

#         # Monte Carlo iterations
#         for it in range(1, n_iterations + 1):
#             end_inv, profit = simulate_one_run(
#                 prices=prices,
#                 costs=costs,
#                 base_utils=base_utils,
#                 init_inventory=init_inventory,
#                 n_customers=n_customers,
#                 rng=rng,
#             )
#             row = [it] + end_inv.tolist() + [profit]
#             writer.writerow(row)
#             rows_written += 1

#     end = time.perf_counter()

#     # Simple summary: mean & std of profit across iterations
#     # (Read back quickly just for the summary; avoids additional memory during loop.)
#     profits = []
#     with open(out_csv, mode="r", newline="") as f:
#         reader = csv.DictReader(f)
#         for r in reader:
#             profits.append(float(r["profit"]))
#     profits = np.array(profits)

#     summary = {
#         "csv_path": out_csv,
#         "iterations": n_iterations,
#         "mean_profit": float(np.mean(profits)) if profits.size else float("nan"),
#         "std_profit": float(np.std(profits, ddof=1)) if profits.size > 1 else float("nan"),
#         "min_profit": float(np.min(profits)) if profits.size else float("nan"),
#         "max_profit": float(np.max(profits)) if profits.size else float("nan"),
#         "runtime_seconds": end - start,
#         "rows_written": rows_written,
#     }
#     return summary

# if __name__ == "__main__":
#     # Parameters set to your initial conditions
#     summary = run_experiment(
#         n_products=10,
#         n_customers=10,
#         init_units=3,
#         prices_list=list(range(6, 16)),  # 6..15
#         cost_value=1.0,
#         utility_list=[6,6,6,9,9,9,12,12,12,15],
#         n_iterations=1000,  # change as needed
#         seed=2026,
#         out_csv="concise_results.csv",
#     )

#     # Print a concise runtime & summary
#     print("---- Run Summary ----")
#     for k, v in summary.items():
#         if k == "runtime_seconds":
#             print(f"{k}: {v:.6f}")
#         else:
#             print(f"{k}: {v}")

# import numpy as np
# import csv
# import time
# from typing import List, Tuple

# # ==============================
# # Dynamic Newsvendor Simulation
# # ==============================
# # Implements sequential customer choice with utility u_i + Gumbel noise,
# # over the in-stock set S(x_t) union {outside option 0}.
# # Results are saved to CSV with header:
# #   iteration,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,profit
# # Runtime is measured with time.perf_counter().

# def gumbel_cdf(x: np.ndarray, mu: float = 0.0, beta: float = 1.0) -> np.ndarray:
#     """Gumbel CDF: F(x) = exp(-exp(-(x-mu)/beta)).
#     Provided for reference; simulation samples via RNG instead.
#     """
#     z = (x - mu) / beta
#     return np.exp(-np.exp(-z))

# def simulate_one_run(
#     prices: np.ndarray,
#     costs: np.ndarray,
#     base_utils: np.ndarray,
#     init_inventory: np.ndarray,
#     n_customers: int,
#     rng: np.random.Generator,
# ) -> Tuple[np.ndarray, float]:
#     """Run a single scenario of the dynamic newsvendor process.

#     Returns:
#         end_inventory: remaining inventory after serving all customers.
#         profit: revenue minus procurement cost of initial inventory.
#     """
#     inventory = init_inventory.astype(int).copy()
#     procurement_cost = float(costs @ init_inventory)

#     revenue = 0.0

#     for _ in range(n_customers):
#         # S(x_t): indices of products with positive inventory
#         in_stock_idx = np.where(inventory > 0)[0]

#         # Candidate set: outside option (-1) plus in-stock products
#         candidates = [-1]
#         if in_stock_idx.size > 0:
#             candidates.extend(in_stock_idx.tolist())

#         # Deterministic utilities
#         det_utils = np.array([
#             0.0 if c == -1 else float(base_utils[c])
#             for c in candidates
#         ])

#         # Add Gumbel noise and choose the argmax
#         eps = rng.gumbel(loc=0.0, scale=1.0, size=len(candidates))
#         total_utils = det_utils + eps
#         choice = candidates[int(np.argmax(total_utils))]

#         # Fulfill if a product was chosen
#         if choice != -1:
#             revenue += float(prices[choice])
#             inventory[choice] -= 1

#     profit = revenue - procurement_cost
#     return inventory, float(profit)

# def run_experiment(
#     n_products: int,
#     n_customers: int,
#     init_units: int,
#     prices_list: List[float],
#     cost_value: float,
#     utility_list: List[float],
#     n_iterations: int,
#     seed: int,
#     out_csv: str,
# ) -> dict:
#     """Run multiple iterations and write results to CSV.

#     CSV header: iteration,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,profit
#     Returns basic summary stats and runtime seconds.
#     """
#     prices = np.array(prices_list, dtype=float)
#     costs = np.full(n_products, float(cost_value))
#     base_utils = np.array(utility_list, dtype=float)
#     init_inventory = np.full(n_products, int(init_units))

#     assert len(prices) == n_products
#     assert len(base_utils) == n_products

#     rng = np.random.default_rng(seed)

#     start = time.perf_counter()

#     header = ["iteration"] + [f"x{i}" for i in range(1, n_products + 1)] + ["profit"]
#     rows_written = 0

#     with open(out_csv, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         for it in range(1, n_iterations + 1):
#             end_inv, profit = simulate_one_run(
#                 prices=prices,
#                 costs=costs,
#                 base_utils=base_utils,
#                 init_inventory=init_inventory,
#                 n_customers=n_customers,
#                 rng=rng,
#             )
#             writer.writerow([it] + end_inv.tolist() + [profit])
#             rows_written += 1

#     end = time.perf_counter()

#     # Summaries
#     profits = []
#     with open(out_csv, mode="r", newline="") as f:
#         reader = csv.DictReader(f)
#         for r in reader:
#             profits.append(float(r["profit"]))
#     profits = np.array(profits)

#     summary = {
#         "csv_path": out_csv,
#         "iterations": n_iterations,
#         "mean_profit": float(np.mean(profits)) if profits.size else float("nan"),
#         "std_profit": float(np.std(profits, ddof=1)) if profits.size > 1 else float("nan"),
#         "min_profit": float(np.min(profits)) if profits.size else float("nan"),
#         "max_profit": float(np.max(profits)) if profits.size else float("nan"),
#         "runtime_seconds": end - start,
#         "rows_written": rows_written,
#     }
#     return summary


# if __name__ == "__main__":
#     # Initial conditions as requested
#     n_products = 10
#     n_customers = 10
#     init_units = 3
#     prices_list = list(range(6, 16))  # [6,7,8,...,15]
#     cost_value = 1.0
#     utility_list = [6,6,6,9,9,9,12,12,12,15]

#     # Experiment controls
#     n_iterations = 1000
#     seed = 2026
#     out_csv = "concise_reults.csv"

#     summary = run_experiment(
#         n_products=n_products,
#         n_customers=n_customers,
#         init_units=init_units,
#         prices_list=prices_list,
#         cost_value=cost_value,
#         utility_list=utility_list,
#         n_iterations=n_iterations,
#         seed=seed,
#         out_csv=out_csv,
#     )

#     # Print summary & runtime
#     print("---- Run Summary ----")
#     for k, v in summary.items():
#         if k == "runtime_seconds":
#             print(f"{k}: {v:.6f}")
#         else:
#             print(f"{k}: {v}")

# import numpy as np
# import csv
# import time

# # ---------------------------------------------------------
# # Problem Parameters (from your specification)
# # ---------------------------------------------------------
# n_products = 10
# n_customers = 10
# n_iterations = 10000

# initial_inventory = np.full(n_products, 3, dtype=int)

# prices = np.arange(6, 16, dtype=float)     # [6,7,...,15]
# costs = np.ones(n_products, dtype=float)   # all 1
# utilities = np.array([6,6,6,9,9,9,12,12,12,15], dtype=float)

# rng = np.random.default_rng(2026)  # reproducible seed


# # ---------------------------------------------------------
# # One iteration of the dynamic newsvendor simulation
# # ---------------------------------------------------------
# def run_single_iteration():
#     inventory = initial_inventory.copy()
#     profit = 0.0

#     for _ in range(n_customers):

#         # Set of in-stock products S(x_t)
#         in_stock = np.where(inventory > 0)[0]

#         # If nothing in stock → only outside option
#         if in_stock.size == 0:
#             continue

#         # Deterministic utilities for in-stock products
#         u = utilities[in_stock]

#         # Gumbel noise for each product + outside option
#         eps_products = rng.gumbel(loc=0.0, scale=1.0, size=in_stock.size)
#         eps_outside = rng.gumbel(loc=0.0, scale=1.0)

#         # Total utilities
#         total_util_products = u + eps_products
#         total_util_outside = eps_outside  # utility of option 0

#         # Choose max over S(x_t) ∪ {0}
#         best_product_idx = np.argmax(total_util_products)
#         best_product_utility = total_util_products[best_product_idx]

#         if best_product_utility > total_util_outside:
#             chosen_global_idx = in_stock[best_product_idx]
#             inventory[chosen_global_idx] -= 1
#             profit += prices[chosen_global_idx] - costs[chosen_global_idx]

#     return inventory, profit


# # ---------------------------------------------------------
# # Main simulation loop
# # ---------------------------------------------------------
# def main():
#     output_file = "concise_results.csv"
#     header = ['iteration'] + [f'x{i+1}' for i in range(n_products)] + ['profit']

#     start = timeConciseStoch.py.time()

#     with open(output_file, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

#         for it in range(1, n_iterations + 1):
#             final_inventory, profit = run_single_iteration()
#             row = [it] + final_inventory.tolist() + [profit]
#             writer.writerow(row)

#     end = time.time()
#     print(f"Completed {n_iterations} iterations")
#     print(f"Runtime: {end - start:.4f} seconds")
#     print(f"Saved to: {output_file}")


# if __name__ == "__main__":
#     main()

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
    profit = 0.0

    for _ in range(n_customers):

        # Set of in-stock products S(x_t)
        in_stock = np.where(inventory > 0)[0]

        # If nothing in stock → customer cannot buy
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

        # Update inventory and profit
        inventory[chosen_global_idx] -= 1
        bought_counts[chosen_global_idx] += 1
        profit += prices[chosen_global_idx] - costs[chosen_global_idx]

    return bought_counts, profit


# ---------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------
def main():
    output_file = "concise_results.csv"
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
    print(f"Completed {n_iterations} iterations")
    print(f"Runtime: {end - start:.4f} seconds")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()