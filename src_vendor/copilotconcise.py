from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence
import math
import copy
import numpy as np

@dataclass
class RetailInstance:
    prices: np.ndarray             # shape: (n,)
    costs: np.ndarray              # shape: (n,)
    salvage: Optional[np.ndarray] = None  # shape: (n,), default zeros
    # Utilities can be:
    #  - a single 2D array of shape (M, n) for one known scenario
    #  - or a list of 2D arrays [(M, n), (M, n), ...] for multiple scenarios
    utilities: Sequence[np.ndarray] = None

    # Constraints (you can set either or both; at least one is recommended)
    capacity: Optional[int] = None           # sum_i x_i <= capacity
    budget: Optional[float] = None           # sum_i costs[i] * x_i <= budget

    # Optional per-item max caps (shape: (n,), entries can be None or int)
    max_per_item: Optional[Sequence[Optional[int]]] = None

    # Tie-breaking: "price" prefers products with higher price on equal utility,
    # "index" prefers lower index, "none" means just utility then index.
    tiebreak: str = "price"

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)
        self.costs = np.asarray(self.costs, dtype=float)
        n = len(self.prices)
        assert self.costs.shape == (n,)
        if self.salvage is None:
            self.salvage = np.zeros(n, dtype=float)
        else:
            self.salvage = np.asarray(self.salvage, dtype = float)
            assert self.salvage.shape == (n,)
        # Normalize utilities into a list of scenarios
        if isinstance(self.utilities, np.ndarray):
            assert self.utilities.ndim == 2 and self.utilities.shape[1] == n
            self.utilities = [self.utilities]
        else:
            assert isinstance(self.utilities, (list, tuple)) and len(self.utilities) > 0
            for U in self.utilities:
                assert isinstance(U, np.ndarray) and U.ndim == 2 and U.shape[1] == n

        # Default per-item caps to None (unbounded) if not provided
        if self.max_per_item is None:
            self.max_per_item = [None] * n
        else:
            assert len(self.max_per_item) == n
            # Convert to ints or None
            self.max_per_item = [None if v is None else int(v) for v in self.max_per_item]

        # Basic sanity: at least one constraint is recommended
        if self.capacity is None and self.budget is None and all(v is None for v in self.max_per_item):
            raise ValueError("Unconstrained inventory would lead to infinite optimal stock. "
                             "Set capacity, budget, or per-item caps.")


def simulate_sales_single_scenario(
    U: np.ndarray,
    x: np.ndarray,
    prices: np.ndarray,
    tiebreak: str = "price"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sequential customer choices for one scenario.

    Args:
        U: (M, n) utilities
        x: (n,) initial inventory (integers)
        prices: (n,) prices
        tiebreak: "price" | "index" | "none"

    Returns:
        sales: (n,) units sold per product
        leftover: (n,) ending inventory
    """
    M, n = U.shape
    inv = x.astype(int).copy()
    sales = np.zeros(n, dtype=int)

    for m in range(M):
        # Consider only in-stock items
        in_stock = np.where(inv > 0)[0]
        if in_stock.size == 0:
            continue
        # Utilities for in-stock items
        u_vals = U[m, in_stock]
        # Choose best if best utility > 0
        max_u = np.max(u_vals)
        if max_u <= 0:
            continue

        # Indices with max utility
        candidates = in_stock[np.where(u_vals == max_u)[0]]

        # Tie-breakers
        if tiebreak == "price" and len(candidates) > 1:
            # pick the highest-priced among candidates
            cand_prices = prices[candidates]
            idx = np.argmax(cand_prices)
            chosen = candidates[idx]
        elif tiebreak == "index" and len(candidates) > 1:
            chosen = int(np.min(candidates))
        else:
            # default: pick lowest index among ties
            chosen = int(np.min(candidates))

        inv[chosen] -= 1
        sales[chosen] += 1

    leftover = inv
    return sales, leftover


def profit_for_x(instance: RetailInstance, x: np.ndarray) -> float:
    """
    Average profit across scenarios for a given x.
    """
    x = x.astype(int)
    total = 0.0
    for U in instance.utilities:
        sales, leftover = simulate_sales_single_scenario(
            U=U,
            x=x,
            prices=instance.prices,
            tiebreak=instance.tiebreak
        )
        revenue = float(np.dot(instance.prices, sales))
        cost = float(np.dot(instance.costs, x))
        salvage = float(np.dot(instance.salvage, leftover))
        total += (revenue - cost + salvage)

    return total / len(instance.utilities)


def is_feasible(instance: RetailInstance, x: np.ndarray) -> bool:
    x = x.astype(int)
    if np.any(x < 0):
        return False
    if instance.capacity is not None and int(np.sum(x)) > instance.capacity:
        return False
    if instance.budget is not None and float(np.dot(instance.costs, x)) > instance.budget + 1e-9:
        return False
    # per-item caps
    for i, cap in enumerate(instance.max_per_item):
        if cap is not None and x[i] > cap:
            return False
    return True


def add_one_feasible_indices(instance: RetailInstance, x: np.ndarray) -> List[int]:
    """
    Return indices i for which adding 1 unit remains feasible.
    """
    feas = []
    for i in range(len(x)):
        if instance.max_per_item[i] is not None and x[i] >= instance.max_per_item[i]:
            continue
        # check adding 1 respects capacity/budget
        if instance.capacity is not None and int(np.sum(x)) + 1 > instance.capacity:
            continue
        if instance.budget is not None and float(np.dot(instance.costs, x) + instance.costs[i]) > instance.budget + 1e-9:
            continue
        feas.append(i)
    return feas


def greedy_marginal_build(instance: RetailInstance, start_x: Optional[np.ndarray] = None, verbose: bool = False
                         ) -> Tuple[np.ndarray, float]:
    """
    Greedily add units with the highest marginal profit until no positive marginal gain
    or constraints block further additions.
    """
    n = len(instance.prices)
    if start_x is None:
        x = np.zeros(n, dtype=int)
    else:
        x = start_x.astype(int).copy()
        if not is_feasible(instance, x):
            raise ValueError("Provided start_x is infeasible.")

    best_profit = profit_for_x(instance, x)

    while True:
        candidates = add_one_feasible_indices(instance, x)
        if not candidates:
            break

        # Evaluate marginal profit for each feasible +1
        best_i = None
        best_delta = 0.0
        for i in candidates:
            x_test = x.copy()
            x_test[i] += 1
            pi = profit_for_x(instance, x_test)
            delta = pi - best_profit
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_i = i

        if best_i is None or best_delta <= 1e-12:
            # no positive marginal improvement
            break

        x[best_i] += 1
        best_profit += best_delta
        if verbose:
            print(f"Add 1 to item {best_i}: Δprofit={best_delta:.4f}, profit={best_profit:.4f}")

    return x, best_profit


def local_swap_improvement(instance: RetailInstance, x: np.ndarray, max_passes: int = 3, verbose: bool = False
                          ) -> Tuple[np.ndarray, float]:
    """
    Try local 1-for-1 swaps (remove 1 from i, add 1 to j) that improve profit and remain feasible.
    If capacity is set but budget is tight (or vice versa), swapping can still help.
    """
    x = x.astype(int).copy()
    base_profit = profit_for_x(instance, x)
    n = len(x)

    for _ in range(max_passes):
        improved = False
        # Try all i->j swaps
        for i in range(n):
            if x[i] <= 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Check per-item cap for j
                if instance.max_per_item[j] is not None and x[j] >= instance.max_per_item[j]:
                    continue
                x_test = x.copy()
                x_test[i] -= 1
                x_test[j] += 1
                # Check capacity: unchanged; Check budget: may change if costs differ
                if not is_feasible(instance, x_test):
                    continue
                pi = profit_for_x(instance, x_test)
                if pi > base_profit + 1e-12:
                    if verbose:
                        print(f"Swap 1 from {i} -> {j}: profit {base_profit:.4f} -> {pi:.4f}")
                    x = x_test
                    base_profit = pi
                    improved = True
        if not improved:
            break
    return x, base_profit


def solve_inventory(instance: RetailInstance, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    End-to-end solver: greedy build + local swap refinement.
    """
    x0, p0 = greedy_marginal_build(instance, verbose=verbose)
    x1, p1 = local_swap_improvement(instance, x0, max_passes=3, verbose=verbose)
    return x1, p1


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example with 10 products and 30 customers (single scenario).
    prices = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
    costs  = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    salvage = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Utilities (M=30 customers, n= products)
    # Positive utility means they'll buy that item if it's their best in-stock option.
    U = np.array([
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ])

    # Constraints: capacity None; budget $1000 total
    instance = RetailInstance(
        prices=prices,
        costs=costs,
        salvage=salvage,
        utilities=U,             # could also pass [U1, U2, ...] to average across scenarios
        capacity=None,
        budget=1000,
        max_per_item=[None, None, None, None, None, None, None, None, None, None],  # no per-item caps
        tiebreak="price"
    )

    #best_x, best_profit = solve_inventory(instance, verbose=True)
    #print("Optimal initial inventory:", best_x)
    #print("Expected profit:", round(best_profit, 4))

    x_test = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    profit = profit_for_x(instance, x_test)
    print("Profit for x=[3,...,3]:", round(profit, 4))

