import numpy as np
from simopt.model import Model
from pydantic import BaseModel


# ------------------------------------------------------------
# Configuration class required by SimOpt
# ------------------------------------------------------------
class DynamNewsConfig(BaseModel):
    num_prod: int
    num_customer: int
    c_utility: list
    mu: float
    init_level: list
    price: list
    cost: list


# ------------------------------------------------------------
# Main Model class
# ------------------------------------------------------------
class DynamNews(Model):

    # REQUIRED by SimOpt
    config_class = DynamNewsConfig

    def __init__(self, fixed_factors):
        super().__init__(fixed_factors=fixed_factors)

        # Extract config values
        cfg = self.config
        self.num_prod = cfg.num_prod
        self.num_customer = cfg.num_customer
        self.c_utility = np.array(cfg.c_utility, dtype=float)
        self.mu = cfg.mu
        self.init_level = np.array(cfg.init_level, dtype=int)
        self.price = np.array(cfg.price, dtype=float)
        self.cost = np.array(cfg.cost, dtype=float)

        # RNG placeholder
        self.rng = None

    # ------------------------------------------------------------
    # SimOpt-required method
    # ------------------------------------------------------------
    def before_replicate(self, rng_list):
        self.rng = rng_list[0]

    # ------------------------------------------------------------
    # Gumbel sampling using MRG32k3a
    # ------------------------------------------------------------
    def sample_gumbel(self, size):
        U = np.array([self.rng.random() for _ in range(size)])
        return -self.mu * np.log(-np.log(U))

    # ------------------------------------------------------------
    # Main replication
    # ------------------------------------------------------------
    def replicate(self):
        inventory = self.init_level.copy()
        profit = 0.0

        for _ in range(self.num_customer):

            in_stock = np.where(inventory > 0)[0]
            if len(in_stock) == 0:
                break

            eps = self.sample_gumbel(self.num_prod)
            utilities = self.c_utility + eps

            j = in_stock[np.argmax(utilities[in_stock])]

            inventory[j] -= 1
            profit += self.price[j] - self.cost[j]

        responses = {"profit": profit}
        gradients = {}

        return responses, gradients