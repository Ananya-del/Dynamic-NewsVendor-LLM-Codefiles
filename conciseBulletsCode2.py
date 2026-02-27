from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


# ==========================
# === USER INPUTS HERE ====
# ==========================

# products: number of products n
products: int = 10              # type: int

# customers: number of sequential customers
customers: int = 30             # type: int

# c_utility: deterministic part u_j for each product j=1..n
c_utility: np.ndarray = np.arange(6, 16)      # type: np.ndarray, shape (products,)

# μ: scale parameter for Gumbel noise ε_jt
mu: float = 1.0                 # type: float

# init_level: initial inventory x_1 = (x_1^1, ..., x_1^n)
init_level: np.ndarray = np.full(10, 3, dtype=int)     # type: np.ndarray, shape (products,)

# price: selling price p_j for each product
price: np.ndarray = np.full(10, 9.0)          # type: np.ndarray, shape (products,)

# cost: unit cost c_j for each product
cost: np.ndarray = np.full(10, 5.0)            # type: np.ndarray, shape (products,)


# ==========================
# === MODEL DEFINITION  ====
# ==========================

@dataclass
class DynamicNewsvendor:
    n_products: int
    u: np.ndarray
    mu: float
    price: np.ndarray
    cost: np.ndarray
    init_inventory: np.ndarray

    def __post_init__(self) -> None:
        assert self.u.shape == (self.n_products,)
        assert self.price.shape == (self.n_products,)
        assert self.cost.shape == (self.n_products,)
        assert self.init_inventory.shape == (self.n_products,)
        self.inventory = self.init_inventory.astype(int).copy()

    def choice_set(self) -> List[int]:
        """S(x_t) = {j : x_jt > 0} ∪ {0}."""
        in_stock = [j for j in range(1, self.n_products + 1) if self.inventory[j - 1] > 0]
        return [0] + in_stock

    def draw_gumbel_noise(self) -> np.ndarray:
        """ε_jt ~ Gumbel(0, μ)."""
        return np.random.gumbel(loc=0.0, scale=self.mu, size=self.n_products)

    def customer_choice(self) -> int:
        """D(x_t, U_t) = argmax_{j in S(x_t)} U_jt."""
        S_xt = self.choice_set()
        eps = self.draw_gumbel_noise()

        utilities = {0: 0.0}
        for j in S_xt:
            if j == 0:
                continue
            utilities[j] = self.u[j - 1] + eps[j - 1]

        return max(S_xt, key=lambda j: utilities[j])

    def step(self) -> Tuple[int, float]:
        """One customer arrival."""
        j = self.customer_choice()
        profit = 0.0
        if j != 0:
            self.inventory[j - 1] -= 1
            profit = self.price[j - 1] - self.cost[j - 1]
        return j, profit

    def simulate(self, n_customers: int) -> Dict[str, np.ndarray]:
        """Simulate sequential customers."""
        choices = np.zeros(n_customers, dtype=int)
        profits = np.zeros(n_customers, dtype=float)
        inventory_history = np.zeros((n_customers + 1, self.n_products), dtype=int)

        inventory_history[0, :] = self.inventory.copy()

        for t in range(n_customers):
            j, prof = self.step()
            choices[t] = j
            profits[t] = prof
            inventory_history[t + 1, :] = self.inventory.copy()

        return {
            "choices": choices,
            "profits": profits,
            "inventory_history": inventory_history,
        }


# ==========================
# === RUNTIME MEASUREMENT ==
# ==========================

if __name__ == "__main__":
    model = DynamicNewsvendor(
        n_products=products,
        u=c_utility,
        mu=mu,
        price=price,
        cost=cost,
        init_inventory=init_level,
    )

    start = time.perf_counter()
    results = model.simulate(n_customers=customers)
    end = time.perf_counter()

    runtime_seconds = end - start

    print("Choices:", results["choices"])
    print("Total profit:", results["profits"].sum())
    print("Final inventory:", results["inventory_history"][-1])
    print(f"Runtime: {runtime_seconds:.6f} seconds")