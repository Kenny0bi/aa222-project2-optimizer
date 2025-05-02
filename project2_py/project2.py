import numpy as np
from scipy.stats import qmc
from project2_py.penalty_method import penalty_method

def optimize(f, g, c, x0, n, count, prob):
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
        dim = len(x0)

        
        if np.all(c(x0) <= 0):
            return x0

        
        x_pm = penalty_method(f, g, c, x0, n, count, prob, max_iters=5)
        if np.all(c(x_pm) <= 0):
            return x_pm

        
        remaining = n - count() - 1
        num_samples = min(2000, max(1, remaining))

        x_best = np.copy(x0)
        best_val = np.inf
        lowest_violation = np.inf

        
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_count = max(1, num_samples // 2)
        try:
            sobol_samples = sampler.random_base2(int(np.ceil(np.log2(sobol_count))))
        except ValueError:
            sobol_samples = sampler.random(n=sobol_count)

        sobol_samples = 2.0 * (sobol_samples - 0.5)
        sobol_samples = x0 + 1.5 * sobol_samples

        for x_try in sobol_samples:
            if count() >= n - 1:
                break
            constraints = c(x_try)
            violation = np.max(constraints)
            if violation <= 0:
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
            elif violation < lowest_violation:
                x_best = x_try
                lowest_violation = violation

        
        for _ in range(num_samples // 2):
            if count() >= n - 1:
                break
            x_try = x0 + 1.5 * np.random.randn(dim)
            constraints = c(x_try)
            violation = np.max(constraints)
            if violation <= 0:
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
            elif violation < lowest_violation:
                x_best = x_try
                lowest_violation = violation

        return x_best

    else:
        return penalty_method(f, g, c, x0, n, count, prob)
