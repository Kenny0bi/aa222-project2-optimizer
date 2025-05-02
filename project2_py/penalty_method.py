import numpy as np

def penalty_method(f, g, c, x0, n, count, max_iters=100, tol=1e-4):
    x = np.array(x0)
    x_best = np.copy(x)
    best_val = np.inf
    rho = 10.0  
    alpha = 0.01
    beta = 0.5

    def is_feasible(x):
        return np.all(c(x) <= 0)

    def penalty_obj(x):
        return f(x) + rho * np.sum(np.maximum(0, c(x))**2)

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

    for outer in range(3):  
        for _ in range(max_iters):
            if count() >= n - 10:  
                return x_best
            grad = penalty_grad(x)
            if np.linalg.norm(grad) < tol:
                break

            t = alpha
            while t > 1e-6:
                if count() >= n - 5:
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
