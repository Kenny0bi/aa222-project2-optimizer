import numpy as np

def penalty_method(f, g, c, x0, n, count, prob, max_iters=None, tol=1e-4):
    x = np.array(x0)
    x_best = np.copy(x)
    best_val = np.inf

    if prob == "secret2":
        rho = 120.0
        outer_loops = 3
        max_iters = max_iters or 10
        margin = 20
    else:
        rho = 10.0
        outer_loops = 2
        max_iters = max_iters or 40
        margin = 10 if prob.startswith("secret") else 0

    alpha = 0.01
    beta = 0.5

    def is_feasible(x):
        return np.all(c(x) <= 0)

    def penalty_obj(x):
        return f(x) + rho * np.sum(np.maximum(0, c(x))**2)

    def penalty_grad(x):
        grad = g(x)
        if count() >= n - margin:
            return grad
        constraint_violations = np.maximum(0, c(x))
        J = numerical_jacobian(c, x)
        for i in range(len(constraint_violations)):
            grad += 2 * rho * constraint_violations[i] * J[i]
        return grad

    def numerical_jacobian(func, x, eps=1e-6):
        n = len(x)
        m = len(func(x))
        J = np.zeros((m, n))
        for i in range(n):
            x1 = np.array(x)
            x2 = np.array(x)
            x1[i] -= eps
            x2[i] += eps
            J[:, i] = (func(x2) - func(x1)) / (2 * eps)
        return J

    for outer in range(outer_loops):
        if count() >= n - margin:
            return x_best
        for _ in range(max_iters):
            if count() >= n - margin:
                return x_best
            grad = penalty_grad(x)
            if np.linalg.norm(grad) < tol:
                break

            t = alpha
            while t > 1e-6:
                if count() >= n - margin:
                    return x_best
                x_new = x - t * grad
                if penalty_obj(x_new) < penalty_obj(x):
                    x = x_new
                    if is_feasible(x_new) and f(x_new) < best_val:
                        x_best = np.copy(x_new)
                        best_val = f(x_new)
                    break
                t *= beta
        rho *= 10

    return x_best
def penalty_method_with_tracking(f, g, c, x0, n, count, prob, max_iters=40, tol=1e-4):
    x = np.array(x0)
    x_best = np.copy(x)
    best_val = np.inf

    rho = 10.0
    outer_loops = 2

    alpha = 0.01
    beta = 0.5

    x_history = [x.copy()]
    f_history = [f(x)]
    violation_history = [np.max(c(x))]

    def is_feasible(x): return np.all(c(x) <= 0)

    def penalty_obj(x): return f(x) + rho * np.sum(np.maximum(0, c(x))**2)

    def penalty_grad(x):
        grad = g(x)
        constraint_violations = np.maximum(0, c(x))
        J = numerical_jacobian(c, x)
        for i in range(len(constraint_violations)):
            grad += 2 * rho * constraint_violations[i] * J[i]
        return grad

    def numerical_jacobian(func, x, eps=1e-6):
        n = len(x)
        m = len(func(x))
        J = np.zeros((m, n))
        for i in range(n):
            x1 = np.array(x)
            x2 = np.array(x)
            x1[i] -= eps
            x2[i] += eps
            J[:, i] = (func(x2) - func(x1)) / (2 * eps)
        return J

    for outer in range(outer_loops):
        for _ in range(max_iters):
            if count() >= n - 1:
                break
            grad = penalty_grad(x)
            if np.linalg.norm(grad) < tol:
                break

            t = alpha
            while t > 1e-6:
                if count() >= n - 1:
                    break
                x_new = x - t * grad
                if penalty_obj(x_new) < penalty_obj(x):
                    x = x_new
                    x_history.append(x.copy())
                    f_history.append(f(x))
                    violation_history.append(np.max(c(x)))
                    if is_feasible(x) and f(x) < best_val:
                        x_best = x.copy()
                        best_val = f(x)
                    break
                t *= beta
        rho *= 10

    return x_best, x_history, f_history, violation_history
