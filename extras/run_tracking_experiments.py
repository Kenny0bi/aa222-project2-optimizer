import numpy as np
from project2_py.penalty_method import penalty_method_with_tracking
from project2_py.hybrid_with_tracking import hybrid_with_tracking


def f(x): return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
def g(x):  
    dfdx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dfdx2 = 200 * (x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])
def c(x): return np.array([(x[0] - 1)**3 - x[1] + 1, x[0] + x[1] - 2])


class Counter:
    def __init__(self): self.calls = 0
    def __call__(self): return self.calls
    def count_f(self): self.calls += 1
    def count_g(self): self.calls += 2
    def count_c(self): self.calls += 1


np.random.seed(42)
initial_conditions = [np.random.uniform(-2, 2, size=2) for _ in range(3)]


penalty_results = []
hybrid_results = []

for x0 in initial_conditions:
    counter1 = Counter()
    counter2 = Counter()

    def f_wrapped(x): counter1.count_f(); return f(x)
    def g_wrapped(x): counter1.count_g(); return g(x)
    def c_wrapped(x): counter1.count_c(); return c(x)

    _, x_hist1, f_hist1, v_hist1 = penalty_method_with_tracking(
        f_wrapped, g_wrapped, c_wrapped, x0, 2000, counter1, "simple2"
    )
    penalty_results.append((x0, x_hist1, f_hist1, v_hist1))

    def f_wrapped2(x): counter2.count_f(); return f(x)
    def g_wrapped2(x): counter2.count_g(); return g(x)
    def c_wrapped2(x): counter2.count_c(); return c(x)

    _, x_hist2, f_hist2, v_hist2 = hybrid_with_tracking(
        f_wrapped2, g_wrapped2, c_wrapped2, x0, 2000, counter2, "simple2"
    )
    hybrid_results.append((x0, x_hist2, f_hist2, v_hist2))


np.savez(
    "tracking_results.npz",
    penalty=np.array(penalty_results, dtype=object),
    hybrid=np.array(hybrid_results, dtype=object),
    initial_conditions=np.array(initial_conditions)
)
