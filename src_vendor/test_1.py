import numpy as np

def simulate_once(
    T,
    n,
    c_utility,
    mu,
    init_level,
    price,
    cost,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # Convert to numpy arrays
    c_utility = np.array(c_utility, dtype=float)
    init_level = np.array(init_level, dtype=int)
    price = np.array(price, dtype=float)
    cost = np.array(cost, dtype=float)

    inventory = init_level.copy()
    sales = np.zeros(n, dtype=int)

    for t in range(T):
        in_stock = (inventory > 0)

        eps = np.random.normal(0, mu, size=n)
        U = c_utility - price + eps

        U_full = np.concatenate(([0], U))
        U_full[1:][~in_stock] = -np.inf

        choice = int(np.argmax(U_full))

        if choice != 0:
            product = choice - 1
            inventory[product] -= 1
            sales[product] += 1

    margin = price - cost
    profit = np.dot(margin, sales)

    return profit, sales


def monte_carlo(
    K=1000,
    T=30,
    n=10,
    c_utility=None,
    mu=1.0,
    init_level=None,
    price=None,
    cost=None
):
    if c_utility is None:
        c_utility = np.arange(6, 6 + n)
    if init_level is None:
        init_level = [3] * n
    if price is None:
        price = [9] * n
    if cost is None:
        cost = [5] * n

    profits = np.zeros(K)
    sales_matrix = np.zeros((K, n), dtype=int)

    for k in range(K):
        profit, sales = simulate_once(
            T=T,
            n=n,
            c_utility=c_utility,
            mu=mu,
            init_level=init_level,
            price=price,
            cost=cost,
            seed=None  # different randomness each run
        )
        profits[k] = profit
        sales_matrix[k] = sales

    return {
        "profits": profits,
        "sales_matrix": sales_matrix,
        "profit_mean": np.mean(profits),
        "profit_std": np.std(profits),
        "avg_sales_per_product": np.mean(sales_matrix, axis=0)
    }


# -------------------------
# Run Monte‑Carlo with your inputs
# -------------------------

result = monte_carlo(
    K=1000,
    T=30,
    n=10,
    c_utility=np.arange(6, 16),
    mu=1.0,
    init_level=[3]*10,
    price=[9]*10,
    cost=[5]*10
)

print("Mean profit:", result["profit_mean"])
print("Std profit:", result["profit_std"])
print("Average sales per product:", result["avg_sales_per_product"])
