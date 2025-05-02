import numpy as np
from scipy.stats import qmc
from project2_py.penalty_method import penalty_method

def optimize(f, g, c, x0, n, count, prob):
    def is_feasible(x): return np.all(c(x) <= 0)
    def max_violation(x): return np.max(c(x))

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
            if is_feasible(x_try):
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
        return x_best

    elif prob == 'secret2':
        dim = len(x0)
        x_best = np.copy(x0)
        best_val = np.inf
        lowest_violation = np.inf
        top_violations = []

        
        for _ in range(20):
            if count() >= n - 1:
                break
            x_try = x0 + 0.1 * np.random.randn(dim)
            if is_feasible(x_try):
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
                return x_best

        
        x_pm = penalty_method(f, g, c, x0, n, count, prob, max_iters=10)
        if is_feasible(x_pm):
            return x_pm

        
        remaining = n - count() - 1
        num_samples = min(2000, max(1, remaining))
        sampler = qmc.Sobol(d=dim, scramble=True)
        try:
            sobol_samples = sampler.random_base2(int(np.ceil(np.log2(num_samples))))
        except:
            sobol_samples = sampler.random(n=num_samples)
        sobol_samples = x0 + 1.0 * (2.0 * (sobol_samples - 0.5))

        for x_try in sobol_samples:
            if count() >= n - 1:
                break
            violation = max_violation(x_try)
            if violation <= 0:
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val
            else:
                top_violations.append((violation, x_try))
                if violation < lowest_violation:
                    x_best = x_try
                    lowest_violation = violation

        
        top_violations.sort(key=lambda tup: tup[0])
        fallback_centers = [x for _, x in top_violations[:3]]

        for center in fallback_centers:
            for _ in range((num_samples // 2) // len(fallback_centers)):
                if count() >= n - 1:
                    break
                x_try = center + 0.25 * np.random.randn(dim)
                violation = max_violation(x_try)
                if violation <= 0:
                    val = f(x_try)
                    if val < best_val:
                        x_best = x_try
                        best_val = val
                elif violation < lowest_violation:
                    x_best = x_try
                    lowest_violation = violation

        
        for center in fallback_centers:
            for step_size in [0.1, 0.05, 0.025]:
                if count() >= n - 1:
                    break
                direction = np.sign(-c(center))  
                perturb = step_size * np.random.randn(dim) + direction
                x_try = center + perturb
                violation = max_violation(x_try)
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
