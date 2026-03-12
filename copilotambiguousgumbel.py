# optimize_inventory_mnl.py

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Union
import numpy as np

# ----------------------
# Constants
# ----------------------
GAMMA_EULER = 0.5772156649015328606  # Euler–Mascheroni constant


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
    # Deterministic utility structure
    beta: float = 1.0       # scales product preferences -> u^j for products
    u0: float = 0.0         # deterministic utility for no-purchase option (j=0)

    # Customer horizon and MC
    num_customers: int = 500
    mc_runs: int = 200
    seed: int = 42

    # MNL (Gumbel) scale: epsilon^j_t ~ Gumbel(loc = -mu*gamma, scale = mu)  -> E[epsilon]=0
    mnl_scale_mu: float = 1.0

    # Constraints
    unit_budget: Optional[int] = None
    procurement_budget: Optional[float] = None

    # Variance reduction
    common_randomness: bool = True


# ----------------------
# Preference Generation (deterministic component u^j for products)
# ----------------------

def default_preference_generator(
    rng: np.random.Generator,
    n_products: int,
    num_customers: int,
    mu: Union[float, np.ndarray] = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Generate a (num_customers x n_products) matrix of base preferences.
    These are converted to deterministic product utilities via beta * pref.
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
# Choice (Random Utility with i.i.d. Gumbel noise) + Simulation
# ----------------------

def mnl_choice(
    utilities: np.ndarray,         # length-n vector of deterministic utilities for products (u^j for j>=1)
    in_stock_mask: np.ndarray,     # boolean mask length-n
    u0: float,                     # deterministic utility for no-purchase (u^0)
    rng: np.random.Generator,
    mu: float = 1.0                # Gumbel scale parameter (epsilon mean = 0 with loc = -mu*gamma)
) -> int:
    """
    Draws i.i.d. Gumbel shocks for every eligible option (products in stock + no-purchase),
    forms U^j = u^j + epsilon^j, and returns the argmax:
      - index in [0, n_products-1] for a purchased product,
      - -1 for no-purchase.

    The specified CDF for epsilon is: P(epsilon <= z) = exp(-exp(-(z/mu + gamma))).
    This equals Gumbel(loc=-mu*gamma, scale=mu).
    """
    eligible_idx = np.where(in_stock_mask)[0]

    # If nothing is in stock, only no-purchase is feasible.
    if eligible_idx.size == 0:
        # Still draw epsilon for no-purchase for completeness (not required)
        return -1

    # Build deterministic utilities for eligible products
    u_prod = utilities[eligible_idx]  # shape (k,)

    # Draw i.i.d. Gumbel shocks with mean 0 by setting loc = -mu*gamma
    # NumPy uses: CDF(x) = exp(-exp(-(x - loc)/scale))
    eps_prod = rng.gumbel(loc=-mu * GAMMA_EULER, scale=mu, size=u_prod.shape[0])
    eps_no  = rng.gumbel(loc=-mu * GAMMA_EULER, scale=mu)

    U_prod = u_prod + eps_prod
    U_no   = u0 + eps_no

    # Compare against no-purchase
    best_prod_idx = int(np.argmax(U_prod))
    if U_no >= U_prod[best_prod_idx]:
        return -1
    else:
        return int(eligible_idx[best_prod_idx])


def simulate_one_run(
    products: List[Product],
    initial_stock: np.ndarray,           # integer vector length-n
    preferences: np.ndarray,             # (T x n) matrix of deterministic preference inputs
    config: Config,
    rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """
    One trajectory. At customer t, deterministic utilities are:
        u^j_t = beta * preferences[t, j] for products j>=1,
        u^0   = config.u0 for no-purchase,
    and random utilities add i.i.d. Gumbel shocks with scale mu = config.mnl_scale_mu.
    """
    n = len(products)
    T = preferences.shape[0]
    stock = initial_stock.copy().astype(int)
    sold = np.zeros(n, dtype=int)

    # Pay procurement up front
    procurement_spend = sum(prod.cost * int(initial_stock[i]) for i, prod in enumerate(products))

    for t in range(T):
        pref_t = preferences[t]         # shape (n,)
        u_det_products = config.beta * pref_t
        in_stock_mask = stock > 0

        choice = mnl_choice(
            utilities=u_det_products,
            in_stock_mask=in_stock_mask,
            u0=config.u0,
            rng=rng,
            mu=config.mnl_scale_mu
        )

        if choice >= 0:
            stock[choice] -= 1
            sold[choice] += 1

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
    Monte Carlo estimate of expected profit with common random numbers (optional).
    """
    n = len(products)
    T = config.num_customers
    runs = config.mc_runs

    if rng_master is None:
        rng_master = np.random.default_rng(config.seed)

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
    # Per-product cap
    if products[k].max_stock is not None and stock[k] + 1 > products[k].max_stock:
        return False

    # Unit budget
    if config.unit_budget is not None and stock.sum() + 1 > config.unit_budget:
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
    n = len(products)
    stock = np.zeros(n, dtype=int)

    rng_master = np.random.default_rng(config.seed)
    base_profit = expected_profit(products, stock, pref_sampler, config, rng_master)

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_k = -1

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
    # Simple demo setup (you can replace this with your specific data)
    products = [
        Product("A", price=12.0, cost=6.0, salvage=1.0, max_stock=200),
        Product("B", price=10.0, cost=4.5, salvage=0.5, max_stock=200),
        Product("C", price=9.0,  cost=4.0, salvage=0.5, max_stock=200),
        Product("D", price=14.0, cost=7.0, salvage=1.5, max_stock=200),
        Product("E", price=8.0,  cost=3.5, salvage=0.5, max_stock=200),
    ]

    config = Config(
        beta=1.0,            # scales deterministic product utilities
        u0=1.0,               # deterministic utility for no-purchase option
        num_customers=15,
        mc_runs=100,
        seed=2026,
        mnl_scale_mu=5.0,     # Gumbel scale (μ). Lower => less randomness; higher => more randomness
        unit_budget=1000,
        procurement_budget=None,
        common_randomness=True
    )

    # Preference generator with heterogeneous means
    mu_vec = np.array([1.0, 0.8, 0.6, 1.2, 0.5])
    sigma = 1.0

    def pref_sampler(rng: np.random.Generator, n: int, T: int) -> np.ndarray:
        return default_preference_generator(rng, n_products=n, num_customers=T, mu=mu_vec, sigma=sigma)

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