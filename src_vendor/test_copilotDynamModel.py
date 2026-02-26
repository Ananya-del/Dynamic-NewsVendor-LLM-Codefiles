"""
Test file for copilotDynamModel.py
Runs 100 replications of the DynamNews model and saves results to CSV.
"""

import csv
import numpy as np

from copilotDynamModel import DynamNews
from mrg32k3a.mrg32k3a import MRG32k3a


def main():
    # ------------------------------------------------------------
    # Define test instance (benchmark)
    # ------------------------------------------------------------
    fixed_factors = {
        "num_prod": 10,
        "num_customer": 30,
        "c_utility": [6 + j for j in range(10)],
        "mu": 1.0,
        "init_level": [3] * 10,
        "price": [9] * 10,
        "cost": [5] * 10,
    }

    # Create model
    model = DynamNews(fixed_factors=fixed_factors)

    # ------------------------------------------------------------
    # Run replications
    # ------------------------------------------------------------
    n_iter = 100
    results = []

    for r in range(n_iter):
        # Create RNG with reproducible 6‑integer seed tuple
        seed_tuple = (r, r + 1, r + 2, r + 3, r + 4, r + 5)
        rng = MRG32k3a(ref_seed=seed_tuple)

        # Prepare model with RNG
        model.before_replicate([rng])

        # Run one replication
        responses, _ = model.replicate()

        # Record results
        results.append({
            "iteration": r + 1,
            **{f"x{j+1}": q for j, q in enumerate(fixed_factors["init_level"])},
            "profit": responses["profit"]
        })

    # ------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------
    csv_file = "copilot_dynamnews_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results of {n_iter} iterations to {csv_file}")


if __name__ == "__main__":
    main()