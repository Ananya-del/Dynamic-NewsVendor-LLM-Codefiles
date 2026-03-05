# test_dynamnews_csv_final.py
import csv
from mrg32k3a.mrg32k3a import MRG32k3a

# Replace with the filename of your simulation code (without .py)
from dynamnews1 import DynamNewsConfig, DynamNews

def main():
    # -------------------------
    # MANUAL INPUTS
    # -------------------------
    num_prod = 10
    num_customer = 10
    init_level = [3] * num_prod
    price = [5]* num_prod  ##[6,7,8,9,10,11,12,13,14,15]
    cost = [1] * num_prod
    c_utility = [0] * num_prod ##[6 + j for j in range(num_prod)]
    mu = 1.0
    n_iterations = 100

    # Create configuration
    config = DynamNewsConfig(
        num_prod=num_prod,
        num_customer=num_customer,
        init_level=init_level,
        price=price,
        cost=cost,
        c_utility=c_utility,
        mu=mu
    )

    # -------------------------
    # INITIALIZE MODEL
    # -------------------------
    model = DynamNews(fixed_factors=config.dict())

    # RNG with reproducible seed
    rng = MRG32k3a(ref_seed=(12345, 12345, 12345, 12345, 12345, 12345))
    model.before_replicate([rng])

    # -------------------------
    # RUN MULTIPLE REPLICATIONS
    # -------------------------
    results = []
    for i in range(n_iterations):
        responses, _ = model.replicate()

        # numsold is returned by the replicate method
        numsold = responses["numsold"]  # array of length num_prod

        # Compose CSV row: iteration, x1..x10, profit
        row = [i + 1] + numsold.tolist() + [responses["profit"]]
        results.append(row)

    # -------------------------
    # SAVE TO CSV
    # -------------------------
    csv_filename = "dynamnews_resultsStoch.csv"
    header = ["iteration"] + [f"x{j+1}" for j in range(num_prod)] + ["profit"]

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"Saved {n_iterations} replications to {csv_filename}")

if __name__ == "__main__":
    main()