import numpy as np
from itertools import product

# ---------------------------------------------------------
# MNL choice probabilities with stockouts
# ---------------------------------------------------------
def mnl_probs(u, mu, available):
    """
    u: deterministic utilities (n,)
    available: boolean mask (n,) for in-stock items
    returns probabilities for [no-purchase, 1..n]
    """
    n = len(u)
    v = np.zeros(n + 1)
    v[1:] = u

    # Mask unavailable products
    masked = v.copy()
    masked[1:][~available] = -np.inf

    exps = np.exp(masked / mu)
    exps[np.isinf(masked)] = 0.0
    return exps / exps.sum()


# ---------------------------------------------------------
# Simulate one sample path of T customers
# ---------------------------------------------------------
def simulate_path(x0, u, mu, prices, costs, T, rng):
    n = len(x0)
    inv = x0.copy()
    revenue = 0.0

    for _ in range(T):
        available = inv > 0
        if not available.any():
            break

        probs = mnl_probs(u, mu, available)
        choice = rng.choice(np.arange(len(probs)), p=probs)

        if choice > 0:
            j = choice - 1
            if inv[j] > 0:
                inv[j] -= 1
                revenue += prices[j]

    cost = np.dot(costs, x0)
    return revenue - cost


# ---------------------------------------------------------
# Monte Carlo expected profit
# ---------------------------------------------------------
def expected_profit(x, u, mu, prices, costs, T, sims=2000):
    rng = np.random.default_rng(0)
    return np.mean([simulate_path(x, u, mu, prices, costs, T, rng) for _ in range(sims)])


# ---------------------------------------------------------
# Constraint placeholder — YOU must specify this
# ---------------------------------------------------------
def feasible(x):
    """
    Replace this with your actual constraint.
    Examples:
      return x.sum() <= 30
      return np.all(x <= 3)
      return np.dot(costs, x) <= 100
    """
    return True   # <-- currently no constraint


# ---------------------------------------------------------
# Brute-force search over inventory levels
# ---------------------------------------------------------
def optimize_inventory(u, mu, prices, costs, T, max_level):
    n = len(u)
    best_x = None
    best_profit = -np.inf

    for x_tuple in product(range(max_level + 1), repeat=n):
        x = np.array(x_tuple)
        if not feasible(x):
            continue

        prof = expected_profit(x, u, mu, prices, costs, T)
        if prof > best_profit:
            best_profit = prof
            best_x = x.copy()

    return best_x, best_profit


# ---------------------------------------------------------
# Example using your parameters
# ---------------------------------------------------------
if __name__ == "__main__":
    n = 10
    T = 30
    u = np.linspace(6, 15, n)      # [6,...,15]
    mu = 1.0
    prices = np.full(n, 9.0)
    costs = np.full(n, 5.0)

    # initial inventory is 3 each, but optimization may change it
    max_level = 6  # search 0..6 for each product (adjust as needed)

    best_x, best_profit = optimize_inventory(u, mu, prices, costs, T, max_level)
    print("Optimal inventory:", best_x)
    print("Expected profit:", best_profit)
