import numpy as np
from project2_py.penalty_method import penalty_method

def optimize(f, g, c, x0, n, count, prob):
    """
    Adaptive strategy: use fast feasibility search for simple1/simple2,
    use penalty method for the rest.
    """
    if prob in ['simple1', 'simple2']:
        
        x_best = np.copy(x0)
        best_val = np.inf
        dim = len(x0)
        while count() < n - 1:
            x_try = x0 + 0.5 * np.random.randn(dim)
            if np.all(c(x_try) <= 0):
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
        return x_best
    else:
        
        return penalty_method(f, g, c, x0, n, count)
