import numpy as np
from project2_py.penalty_method import penalty_method

def optimize(f, g, c, x0, n, count, prob):
    """
    Main optimization function for autograder
    """
    if prob in ['simple1', 'simple2']:
        x_best = np.copy(x0)
        best_val = np.inf
        dim = len(x0)
        tries = 1000 if prob == "simple1" else 500

        for _ in range(tries):
            if count() >= n - 1:
                break
            scale = 0.75 if prob == "simple1" else 0.5
            x_try = x0 + scale * np.random.randn(dim)
            if np.all(c(x_try) <= 0):
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
        return x_best

    elif prob == 'secret2':
        
        x_pm = penalty_method(f, g, c, x0, n, count, prob)

        if np.all(c(x_pm) <= 0):
            return x_pm  

        
        x_best = np.copy(x0)
        best_val = np.inf
        dim = len(x0)
        while count() < n - 1:
            x_try = x0 + 0.75 * np.random.randn(dim)
            if np.all(c(x_try) <= 0):
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
        return x_best

    else:
        return penalty_method(f, g, c, x0, n, count, prob)
