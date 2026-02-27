import numpy as np
import itertools

# ------------------------------------------------------------
# 1. SIMULATION OF CUSTOMER CHOICE
# ------------------------------------------------------------

def simulate_profit(x, prices, costs, U):
    """
    Simulates sequential customer choices given inventory x.
    U[t,j] includes j=0 (no-purchase) with U[t,0] = 0.
    """
    x = np.array(x, dtype=int)
    prices = np.array(prices)
    costs = np.array(costs)
    U = np.array(U)

    T, cols = U.shape
    n = len(x)
    assert cols == n + 1

    inventory = x.copy()
    sales = np.zeros(n, dtype=int)

    for t in range(T):
        u_t = U[t].copy()

        # Make out-of-stock products unavailable
        for j in range(1, n + 1):
            if inventory[j - 1] <= 0:
                u_t[j] = -np.inf

        choice = int(np.argmax(u_t))

        if choice != 0:
            j = choice - 1
            inventory[j] -= 1
            sales[j] += 1

    profit = np.dot(prices - costs, sales)
    return profit, sales


# ------------------------------------------------------------
# 2. OPTIONAL: BRUTE-FORCE OPTIMIZATION OVER INVENTORY
# ------------------------------------------------------------

def brute_force_optimize(prices, costs, U, max_inv):
    n = len(prices)
    ranges = [range(max_inv + 1)] * n

    best_profit = -np.inf
    best_x = None

    for x in itertools.product(*ranges):
        profit, _ = simulate_profit(x, prices, costs, U)
        if profit > best_profit:
            best_profit = profit
            best_x = x

    return best_x, best_profit


# ------------------------------------------------------------
# 3. GENERATE YOUR INSTANCE (10 PRODUCTS, 30 CUSTOMERS)
# ------------------------------------------------------------

n = 10
T = 30

# Prices and costs
prices = [9] * n
costs = [5] * n

# Initial inventory (all 3)
x0 = [3] * n

# Utility generation:
# You said utilities are in the range 6–15 with μ = 1.0.
# We'll generate exponential(μ) and shift/scale to [6,15].
np.random.seed(0)

raw = np.random.exponential(scale=1.0, size=(T, n))
U_products = 6 + 9 * raw / raw.max()   # scale to [6,15]

# Add no-purchase option U^0 = 0
U = np.hstack([np.zeros((T,1)), U_products])

# ------------------------------------------------------------
# 4. RUN SIMULATION
# ------------------------------------------------------------

profit, sales = simulate_profit(x0, prices, costs, U)

print("Initial inventory:", x0)
print("Sales by product:", sales)
print("Profit:", profit)

# ------------------------------------------------------------
# 5. OPTIONAL: FIND BEST INVENTORY (small search only)
# ------------------------------------------------------------
# best_x, best_profit = brute_force_optimize(prices, costs, U, max_inv=5)
# print("Optimal inventory:", best_x)
# print("Optimal profit:", best_profit)
