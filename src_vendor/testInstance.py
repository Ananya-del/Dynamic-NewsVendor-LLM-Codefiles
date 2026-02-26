# testInstanceCSV.py
import sys
import numpy as np
import csv

# Fix Python path to find simopt package
sys.path.append(r"C:\Users\4anan\OneDrive\Documents\Spring 2026\Research\simopt-master\simopt-master")

from simopt.models.dynamnews import DynamNews
from mrg32k3a.mrg32k3a import MRG32k3a

def main():
    # Define test instance (benchmark)
    fixed_factors = {
        "num_prod": 10,
        "num_customer": 30,
        "c_utility": [6 + j for j in range(10)],
        "mu": 1.0,
        "init_level": [3] * 10,
        "price": [9] * 10,
        "cost": [5] * 10,
    }

    model = DynamNews(fixed_factors=fixed_factors)

    n_iter = 100  # number of iterations
    results = []

    for r in range(n_iter):
        # create RNG with reproducible seed
        seed_tuple = (r, r+1, r+2, r+3, r+4, r+5)
        rng = MRG32k3a(ref_seed=seed_tuple)

        # prepare model with RNG
        model.before_replicate([rng])
        responses, _ = model.replicate()

        # record initial inventory and profit
        results.append({
            "iteration": r + 1,
            **{f"x{j+1}": q for j, q in enumerate(fixed_factors["init_level"])},
            "profit": responses["profit"]
        })

    # Save to CSV
    csv_file = "dynamnews_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results of {n_iter} iterations to {csv_file}")

if __name__ == "__main__":
    main()
# # testInstance.py
# import sys
# import os
# import numpy as np

# # -----------------------------
# # Fix Python path to find simopt package
# # -----------------------------
# sys.path.append(r"C:\Users\4anan\OneDrive\Documents\Spring 2026\Research\simopt-master\simopt-master")

# # -----------------------------
# # Imports
# # -----------------------------
# from simopt.models.dynamnews import DynamNews
# from mrg32k3a.mrg32k3a import MRG32k3a

# # -----------------------------
# # Main function
# # -----------------------------
# def main():
#     # ---------------------------
#     # Define test instance (benchmark)
#     # ---------------------------
#     fixed_factors = {
#         "num_prod": 10,
#         "num_customer": 30,
#         "c_utility": [6 + j for j in range(10)],
#         "mu": 1.0,
#         "init_level": [3] * 10,
#         "price": [9] * 10,
#         "cost": [5] * 10,
#     }

#     # Create model
#     model = DynamNews(fixed_factors=fixed_factors)

#     # ---------------------------
#     # Run 1000 replications
#     # ---------------------------
#     n_rep = 1000
#     profits = []

#     for r in range(n_rep):
#         # create RNG with a reproducible seed (6-integer tuple)
#         seed_tuple = (r, r + 1, r + 2, r + 3, r + 4, r + 5)
#         rng = MRG32k3a(ref_seed=seed_tuple)

#         # prepare model with RNG
#         model.before_replicate([rng])

#         # run one replication
#         responses, _ = model.replicate()
#         profits.append(responses["profit"])

#     profits = np.array(profits)
#     mean_profit = profits.mean()
#     std_error = profits.std(ddof=1) / np.sqrt(n_rep)

#     # ---------------------------
#     # Print results
#     # ---------------------------
#     print("Benchmark Test Instance")
#     print("----------------------")
#     print("Number of replications:", n_rep)
#     print("Mean profit:", mean_profit)
#     print("Standard error:", std_error)


# # -----------------------------
# # Run main
# # -----------------------------
# if __name__ == "__main__":
#     main()