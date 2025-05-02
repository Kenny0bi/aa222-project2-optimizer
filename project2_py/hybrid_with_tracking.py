import numpy as np
from scipy.stats import qmc
from project2_py.penalty_method import penalty_method_with_tracking

def hybrid_with_tracking(f, g, c, x0, n, count, prob):
    dim = len(x0)
    x_best = np.copy(x0)
    best_val = np.inf
    lowest_violation = np.inf

    x_history = [x0.copy()]
    f_history = [f(x0)]
    violation_history = [np.max(c(x0))]

    def is_feasible(x): return np.all(c(x) <= 0)
    def max_violation(x): return np.max(c(x))

    # Step 1: Slight x0 perturbation
    for _ in range(20):
        if count() >= n - 1: break
        x_try = x0 + 0.1 * np.random.randn(dim)
        violation = max_violation(x_try)
        x_history.append(x_try)
        f_history.append(f(x_try))
        violation_history.append(violation)
        if violation <= 0:
            val = f(x_try)
            if val < best_val:
                x_best = x_try
                best_val = val
            return x_best, x_history, f_history, violation_history

    # Step 2: Penalty method
    x_pm, x_hist_pm, f_hist_pm, v_hist_pm = penalty_method_with_tracking(f, g, c, x0, n, count, prob)
    x_history += x_hist_pm[1:]
    f_history += f_hist_pm[1:]
    violation_history += v_hist_pm[1:]
    if is_feasible(x_pm):
        return x_pm, x_history, f_history, violation_history

    # Step 3: Sobol sampling
    num_samples = min(500, n - count() - 1)
    sampler = qmc.Sobol(d=dim, scramble=True)
    sobol_samples = sampler.random_base2(int(np.ceil(np.log2(num_samples))))
    sobol_samples = x0 + 1.0 * (2.0 * (sobol_samples - 0.5))
    top_violations = []

    for x_try in sobol_samples:
        if count() >= n - 1: break
        violation = max_violation(x_try)
        x_history.append(x_try)
        f_history.append(f(x_try))
        violation_history.append(violation)
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

    # Step 4: Local refinement near best violators
    top_violations.sort(key=lambda t: t[0])
    fallback_centers = [x for _, x in top_violations[:3]]
    for center in fallback_centers:
        for _ in range(30):
            if count() >= n - 1: break
            x_try = center + 0.25 * np.random.randn(dim)
            violation = max_violation(x_try)
            x_history.append(x_try)
            f_history.append(f(x_try))
            violation_history.append(violation)
            if violation <= 0:
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val

    # Step 5: Directional constraint-nudging
    for center in fallback_centers:
        for step_size in [0.1, 0.05, 0.025]:
            if count() >= n - 1: break
            direction = np.sign(-c(center))
            perturb = step_size * np.random.randn(dim) + direction
            x_try = center + perturb
            violation = max_violation(x_try)
            x_history.append(x_try)
            f_history.append(f(x_try))
            violation_history.append(violation)
            if violation <= 0:
                val = f(x_try)
                if val < best_val:
                    x_best = x_try
                    best_val = val

    return x_best, x_history, f_history, violation_history
