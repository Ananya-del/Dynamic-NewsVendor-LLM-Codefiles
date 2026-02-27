import numpy as np

# -----------------------------
# Parameters
# -----------------------------
num_products = 10
num_customers = 30

initial_inventory = np.array([3] * num_products)
price = 9
cost = 5
profit_per_unit = price - cost

# Generate utilities for each customer-product pair
# Utilities uniformly from 6 to 15 (inclusive)
utilities = np.random.randint(6, 16, size=(num_customers, num_products))

# Track results
inventory = initial_inventory.copy()
sales = np.zeros(num_products, dtype=int)
customer_choices = []

# -----------------------------
# Simulation
# -----------------------------
for c in range(num_customers):
    # Compute net utility = utility - price
    net_util = utilities[c] - price

    # Mask out products that are out of stock
    available_mask = inventory > 0
    net_util_available = np.where(available_mask, net_util, -np.inf)

    # Customer chooses the product with highest net utility if positive
    best_product = np.argmax(net_util_available)
    best_value = net_util_available[best_product]

    if best_value > 0:
        # Customer buys this product
        inventory[best_product] -= 1
        sales[best_product] += 1
        customer_choices.append(best_product)
    else:
        # Customer buys nothing
        customer_choices.append(None)

# -----------------------------
# Results
# -----------------------------
total_sales = np.sum(sales)
total_profit = total_sales * profit_per_unit

print("=== Simulation Results ===")
print(f"Initial inventory per product: {initial_inventory}")
print(f"Final inventory per product:   {inventory}")
print(f"Units sold per product:        {sales}")
print(f"Total units sold:              {total_sales}")
print(f"Total profit:                  {total_profit}")
print("\nCustomer choices (None = no purchase):")
print(customer_choices)
