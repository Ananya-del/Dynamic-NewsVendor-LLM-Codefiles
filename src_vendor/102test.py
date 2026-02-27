import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Product:
    name: str
    price: float
    cost: float
    inventory: int

    @property
    def margin(self):
        return self.price - self.cost


@dataclass
class Customer:
    utilities: np.ndarray  # utility for each product
    no_purchase_utility: float = 0.0

    def choose(self, available_mask: np.ndarray) -> Optional[int]:
        """
        available_mask: boolean array indicating which products are in stock.
        Returns index of chosen product or None.
        """
        # Mask unavailable products
        masked_utilities = np.where(available_mask, self.utilities, -np.inf)

        # Compare best product utility vs no-purchase
        best_product = np.argmax(masked_utilities)
        best_utility = masked_utilities[best_product]

        if best_utility >= self.no_purchase_utility:
            return best_product
        return None


@dataclass
class SimulationResult:
    choices: List[Optional[int]]
    profit: float
    remaining_inventory: List[int]


class RetailSimulation:
    def __init__(self, products: List[Product]):
        self.products = products

    def run(self, customers: List[Customer]) -> SimulationResult:
        choices = []
        total_profit = 0.0

        for customer in customers:
            available_mask = np.array([p.inventory > 0 for p in self.products])
            choice = customer.choose(available_mask)
            choices.append(choice)

            if choice is not None:
                product = self.products[choice]
                product.inventory -= 1
                total_profit += product.margin

        remaining_inventory = [p.inventory for p in self.products]
        return SimulationResult(choices, total_profit, remaining_inventory)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Define products
    products = [
        Product("A", price=9, cost=5, inventory=3),
        Product("B", price=9, cost=5, inventory=3),
        Product("C", price=9, cost=5, inventory=3),
        Product("D", price=9, cost=5, inventory=3),
        Product("E", price=9, cost=5, inventory=3),
        Product("F", price=9, cost=5, inventory=3),
        Product("G", price=9, cost=5, inventory=3),
        Product("H", price=9, cost=5, inventory=3),
        Product("I", price=9, cost=5, inventory=3),
        Product("J", price=9, cost=5, inventory=3)
    ]

    # Generate customers with random utilities
    rng = np.random.default_rng(42)
    customers = [
        Customer(utilities=rng.normal(loc=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15], scale=1))  # mean utilities differ by product
        for _ in range(20)
    ]

    sim = RetailSimulation(products)
    result = sim.run(customers)

    print("Choices:", result.choices)
    print("Total Profit:", result.profit)
    print("Remaining Inventory:", result.remaining_inventory)
