# optimize_inventory_mnl.py

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Union
import numpy as np

# ----------------------
# Data Structures
# ----------------------

@dataclass
class Product:
    name: str
    price: float      # selling price p_i
    cost: float       # procurement cost c_i (paid for all ordered units)
    salvage: float = 0.0  # salvage value v_i for leftover units
    max_stock: Optional[int] = None  # optional per-product cap


@dataclass
class Config:
    beta: float = 1.0           # choice sensitivity (higher => more deterministic by preference)
    u0: float = 0.0             # utility for no-purchase option
    num_customers: int = 500    # number of customer arrivals
    mc_runs: int = 200          # Monte Carlo replications for expected values
    seed: int = 42              # RNG seed for reproducibility
    # Optional constraints on initial inventory decision:
    unit_budget: Optional[int] = None         # total number of units allowed, sum(s_i) <= unit_budget
    procurement_budget: Optional[float] = None # total procurement spend allowed, sum(c_i * s_i) <= procurement_budget
    # Performance controls:
    common_randomness: bool = True            # use the same random streams when comparing candidate solutions


# ----------------------
# Preference Generation
# ----------------------

def default_preference_generator(
    rng: np.random.Generator,
    n_products: int,
    num_customers: int,
    mu: Union[float, np.ndarray] = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Generate a (num_customers x n_products) matrix of preferences.
    By default, each customer's preference for product i is Normal(mu_i, sigma^2).
    - mu can be a scalar (same mean for all products) or a 1D array of length n_products.
    """
    if np.isscalar(mu):
        mu_vec = np.full(n_products, float(mu))
    else:
        mu_vec = np.asarray(mu, dtype=float)
        if mu_vec.shape != (n_products,):
            raise ValueError("mu must be scalar or length n_products.")
    prefs = rng.normal(loc=mu_vec, scale=sigma, size=(num_customers, n_products))
    return prefs


# ----------------------
# Choice + Simulation
# ----------------------

def mnl_choice(
    utilities: np.ndarray,         # length-n vector for in-stock products' utilities (beta * pref)
    in_stock_mask: np.ndarray,     # boolean mask length-n
    u0: float,                     # no-purchase utility
    rng: np.random.Generator
) -> int:
    """
    Returns chosen index in [0, n_products-1], or -1 for no-purchase.
    Only in-stock items are eligible.
    """
    eligible = np.where(in_stock_mask)[0]
    if eligible.size == 0:
        # Only no-purchase available
        return -1

    # Compute MNL probabilities among eligible + no-purchase
    util = utilities.copy()
    # For out-of-stock, set to -inf (not chosen)
    util[~in_stock_mask] = -np.inf

    # Compute exp utilities safely
    exp_utils = np.zeros_like(util)
    mask_finite = np.isfinite(util)
    exp_utils[mask_finite] = np.exp(util[mask_finite])

    # Include no-purchase as an option
    exp_u0 = np.exp(u0)

    denom = exp_u0 + exp_utils.sum()
    # Probabilities for products:
    probs_prod = exp_utils / denom
    # No-purchase probability:
    prob_no = exp_u0 / denom

    # Sample choice
    # Construct categorical over [products..., no-purchase]
    probs = np.append(probs_prod, prob_no)
    choice = rng.choice(len(probs), p=probs)
    if choice == len(probs) - 1:
        return -1  # no-purchase
    # map back to product index
    return choice


def simulate_one_run(
    products: List[Product],
    initial_stock: np.ndarray,           # integer vector length-n
    preferences: np.ndarray,             # (T x n) matrix
    config: Config,
    rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """
    Run a single trajectory. Returns:
      - sold_counts: vector length-n of units sold
      - profit: realized profit for the trajectory
    Profit = sum_i sold_i * price_i - sum_i initial_stock_i * cost_i + sum_i leftover_i * salvage_i
    """
    n = len(products)
    T = preferences.shape[0]
    stock = initial_stock.copy().astype(int)
    sold = np.zeros(n, dtype=int)

    # Pay procurement up front
    procurement_spend = sum(prod.cost * int(initial_stock[i]) for i, prod in enumerate(products))

    for t in range(T):
        pref_t = preferences[t]  # shape (n,)
        # Utilities = beta * preference
        utilities = config.beta * pref_t
        in_stock_mask = stock > 0

        choice = mnl_choice(utilities, in_stock_mask, config.u0, rng)
        if choice >= 0:
            # Fulfill sale
            stock[choice] -= 1
            sold[choice] += 1
            # If stock reaches 0, item disappears for next customers

    # Leftovers salvage
    leftovers = stock
    revenue = sum(products[i].price * sold[i] for i in range(n))
    salvage = sum(products[i].salvage * leftovers[i] for i in range(n))
    profit = revenue - procurement_spend + salvage

    return sold, profit


def expected_profit(
    products: List[Product],
    initial_stock: np.ndarray,
    pref_sampler: Callable[[np.random.Generator, int, int], np.ndarray],
    config: Config,
    rng_master: Optional[np.random.Generator] = None
) -> float:
    """
    Monte Carlo estimate of expected profit for a given stock vector.
    If config.common_randomness is True, uses fixed seeds across calls for variance reduction.
    """
    n = len(products)
    T = config.num_customers
    runs = config.mc_runs

    if rng_master is None:
        rng_master = np.random.default_rng(config.seed)

    # Common random numbers: pre-generate seeds for each run.
    seeds = rng_master.integers(1, 2**31 - 1, size=runs) if config.common_randomness else None

    profits = []
    for r in range(runs):
        rng_run = np.random.default_rng(seeds[r]) if seeds is not None else np.random.default_rng()
        prefs = pref_sampler(rng_run, n, T)
        _, prof = simulate_one_run(products, initial_stock, prefs, config, rng_run)
        profits.append(prof)

    return float(np.mean(profits))


# ----------------------
# Feasibility + Greedy Optimizer
# ----------------------

def feasible_to_add(
    products: List[Product],
    stock: np.ndarray,
    k: int,
    config: Config
) -> bool:
    """
    Check if adding one unit of product k keeps constraints satisfied.
    """
    # Per-product cap
    if products[k].max_stock is not None and stock[k] + 1 > products[k].max_stock:
        return False

    # Unit budget
    if config.unit_budget is not None:
        if stock.sum() + 1 > config.unit_budget:
            return False

    # Procurement budget
    if config.procurement_budget is not None:
        current_spend = float(np.dot(stock, np.array([p.cost for p in products])))
        if current_spend + products[k].cost > config.procurement_budget + 1e-9:
            return False

    return True


def greedy_optimize_inventory(
    products: List[Product],
    pref_sampler: Callable[[np.random.Generator, int, int], np.ndarray],
    config: Config
) -> Tuple[np.ndarray, float]:
    """
    Start from zero stock. Iteratively add one unit of the product with the largest
    positive marginal expected profit, respecting constraints. Stops when no positive
    marginal gains remain or constraints are binding.
    Returns (best_stock_vector, expected_profit).
    """
    n = len(products)
    stock = np.zeros(n, dtype=int)

    # Prepare a master RNG for common random numbers
    rng_master = np.random.default_rng(config.seed)

    # Compute baseline expected profit
    base_profit = expected_profit(products, stock, pref_sampler, config, rng_master)

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_k = -1

        # Evaluate marginal gain for each product
        for k in range(n):
            if not feasible_to_add(products, stock, k, config):
                continue

            candidate = stock.copy()
            candidate[k] += 1
            cand_profit = expected_profit(products, candidate, pref_sampler, config, rng_master)
            gain = cand_profit - base_profit

            if gain > best_gain + 1e-9:
                best_gain = gain
                best_k = k

        if best_k >= 0 and best_gain > 0:
            # Accept the best move
            stock[best_k] += 1
            base_profit += best_gain
            improved = True
        else:
            break

    return stock, base_profit


# ----------------------
# Example usage
# ----------------------

def example():
    # 10 products, each price=9, cost=5, salvage=0, max_stock=3
    products = [
        Product(f"P{i+1}", price=9.0, cost=5.0, salvage=0.0, max_stock=3)
        for i in range(10)
    ]

    # Configuration for 30 customers
    config = Config(
        beta=1.0,             # keep as-is unless you want more deterministic choices
        u0=0.0,               # no-purchase utility; with positive utilities, most will purchase
        num_customers=30,     # exactly 30 customers
        mc_runs=1,          # Monte Carlo runs
        seed=2026,
        unit_budget=None,           # no global unit cap (per-product cap is already 3)
        procurement_budget=None,    # no procurement budget
        common_randomness=True
    )

    # For every customer, utilities for products are [6,7,8,9,10,11,12,13,14,15]
    base_util = np.arange(6, 16, dtype=float)  # length 10

    def pref_sampler(rng: np.random.Generator, n: int, T: int) -> np.ndarray:
        # Deterministic preferences: T x n where each row is base_util
        # (We still follow the required signature.)
        if n != 10:
            raise ValueError("This setup expects exactly 10 products.")
        return np.tile(base_util, (T, 1))

    # Run optimizer
    best_stock, best_exp_profit = greedy_optimize_inventory(products, pref_sampler, config)

    print("\nRecommended initial inventory (units):")
    for i, prod in enumerate(products):
        print(f"  {prod.name}: {best_stock[i]}")

    total_units = int(best_stock.sum())
    total_proc_cost = float(np.dot(best_stock, np.array([p.cost for p in products])))
    print(f"\nTotal units: {total_units}")
    print(f"Total procurement cost: ${total_proc_cost:,.2f}")
    print(f"Estimated expected profit: ${best_exp_profit:,.2f}\n")


if __name__ == "__main__":
    example()
