# ===========================
# Dynamic Newsvendor Simulator
# With Gumbel utilities and in-stock choice set S(x_t) ∪ {0}
# ===========================
# This script simulates sequential customer choices where each customer chooses
# the highest utility option among in-stock products S(x_t) plus a no-purchase option {0}.
# Utilities are modeled as u_i + ε_i with ε_i ~ Gumbel(0, μ). Inventory is discrete (integers)
# and never goes negative; only in-stock items can be chosen.
#
# HOW TO USE:
# - Edit the INPUTS block at the bottom to set:
#   products, customers, p_utility (per-product utilities), c_utility (outside option),
#   viscosity (μ), init_level (integer stocks), price, cost, and seed (optional).
# - Run the file with Python 3:  python dynamic_newsvendor.py

import math
import random
from typing import Dict, List, Any


# --------- Gumbel tools ----------
def gumbel_sample(scale_mu: float = 1.0, loc: float = 0.0, rng: random.Random = None) -> float:
    """
    Draw a single Gumbel(loc, scale_mu) variate using inverse CDF:
        ε = loc - scale_mu * ln(-ln(U)),  where U ~ Uniform(0,1)
    scale_mu > 0 controls randomness (often called the 'temperature' in discrete choice).
    """
    if rng is None:
        rng = random
    u = rng.random()
    # Guard against log(0) if rng returns exactly 0.0 (practically never, but safe)
    u = min(max(u, 1e-16), 1 - 1e-16)
    return loc - scale_mu * math.log(-math.log(u))


def gumbel_cdf(x: float, loc: float = 0.0, scale_mu: float = 1.0) -> float:
    """
    CDF of Gumbel(loc, scale_mu): F(x) = exp(-exp(-(x - loc)/scale_mu))
    """
    z = (x - loc) / scale_mu
    return math.exp(-math.exp(-z))


# --------- Helpers ----------
def map_by_products(products: List[str], values: List[float], name: str) -> Dict[str, float]:
    """
    Coerce a list of values into a product-keyed dict with validation.
    """
    if len(values) != len(products):
        raise ValueError(f"Length mismatch: {name} has {len(values)} values but products has {len(products)}.")
    return {p: float(v) for p, v in zip(products, values)}


def map_ints_by_products(products: List[str], values: List[int], name: str) -> Dict[str, int]:
    """
    Coerce a list of integers into a product-keyed dict with validation and non-negativity.
    """
    if len(values) != len(products):
        raise ValueError(f"Length mismatch: {name} has {len(values)} values but products has {len(products)}.")
    inv = {}
    for p, v in zip(products, values):
        # Accept ints or floats that are whole numbers
        if isinstance(v, float) and not float(v).is_integer():
            raise ValueError(f"{name} must be integers; got non-integer {v} for product {p}.")
        v_int = int(v)
        if v_int < 0:
            raise ValueError(f"{name} must be non-negative; got {v_int} for product {p}.")
        inv[p] = v_int
    return inv


def in_stock_set(inventory: Dict[str, int]) -> List[str]:
    """Return the list S(x_t) of products currently in stock (positive inventory)."""
    return [p for p, qty in inventory.items() if qty > 0]


# --------- Core simulation ----------
def simulate_dynamic_newsvendor(
    products: List[str],
    customers: int,
    p_utility: List[float],         # per-product base utilities u_i
    c_utility: float,               # outside option base utility (for "0")
    viscosity_mu: float,            # Gumbel scale μ (controls randomness)
    init_level: List[int],          # initial inventory per product (integers)
    price: List[float],
    cost: List[float],
    seed: int = None,
    keep_trace: bool = False
) -> Dict[str, Any]:
    
    # ===========================
# >>>>>>>>>>>  INPUTS  <<<<<<<<<<<
# Edit this block with your scenario
    if __name__ == "__main__":
    # products: list/array-like identifiers (10 products)
        products = [f"P{i}" for i in range(1, 11)]  # ['P1', 'P2', ..., 'P10']

    # customers: total sequential arrivals
        customers = 30

    # p_utility: per-product base utilities u_i (same order as `products`)
    # Interpreting your "c_utility = [6,...,15]" as *per-product* utilities.
        p_utility = list(range(6, 16))  # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # c_utility: outside (no-purchase) base utility (scalar)
    # Choose a baseline; adjust as needed to increase/decrease no-purchase propensity.
        c_utility = 0.0

    # viscosity (μ): Gumbel scale parameter (>0). Higher μ => more randomness.
        viscosity = 1.0

    # init_level: initial inventory per product (integers) — all 3s
        init_level = [3] * len(products)

    # price and cost per product (can be int or float), same length/order as `products`
        price = [9.0] * len(products)
        cost  = [5.0] * len(products)

    # Optional: set a seed for reproducibility
        seed = 42

    # Run the simulation
        results = simulate_dynamic_newsvendor(
            products=products,
            customers=customers,
            p_utility=p_utility,
            c_utility=c_utility,
            viscosity_mu=viscosity,
            init_level=init_level,
            price=price,
            cost=cost,
            seed=seed,
            keep_trace=False  # set True to get period-by-period details
        )

    # Pretty-print results
        print("\n=== Simulation Results ===")
        print(f"Revenue: {results['revenue']:.2f}")
        print(f"Cost:    {results['cost']:.2f}")
        print(f"Profit:  {results['profit']:.2f}")
        print("\nUnits sold per product:")
        for p in products:
            print(f"  {p}: {results['units_sold'][p]}")
        print("\nRemaining inventory:")
        for p in products:
            print(f"  {p}: {results['remaining_inventory'][p]}")
        print(f"\nNo-purchase count (outside option '0'): {results['no_purchase_count']}")
        print(f"Times system empty (S(x_t) = ∅): {results['empty_system_count']}")
        print("\nChoice counts (including '0'):")
        for k, v in results['choice_counts'].items():
            print(f"  {k}: {v}")
    """
    Simulate sequential customer choices with Gumbel utilities over the in-stock set S(x_t),
    unioned with the outside (no-purchase) option {0}. Inventory is discrete and never negative.

    Args:
        products: list of product identifiers (strings recommended)
        customers: number of sequential arrivals (int)
        p_utility: list of per-product base utilities u_i
        c_utility: outside-option utility constant (scalar)
        viscosity_mu: Gumbel scale parameter μ > 0 (higher → more randomness)
        init_level: initial inventory per product (ints, non-negative)
        price, cost: lists aligned with products
        seed: optional RNG seed for reproducibility
        keep_trace: if True, include period-by-period trace
        """