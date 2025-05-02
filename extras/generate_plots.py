import numpy as np
import matplotlib.pyplot as plt

# Objective and constraints for simple1 and simple2
def f1(x): return -x[0]*x[1] + 2 / (3**0.5)
def c1(x): return np.array([x[0] + x[1]**2 - 1, -x[0] - x[1]])

def f2(x): return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def c2(x): return np.array([(x[0] - 1)**3 - x[1] + 1, x[0] + x[1] - 2])

def plot_contour(f, c, x_histories, title, xlim=(-3, 3), ylim=(-3, 3), filename=None):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    C1 = np.array([[c([x_, y_])[0] for x_, y_ in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    C2 = np.array([[c([x_, y_])[1] for x_, y_ in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=30, cmap="viridis")
    ax.contour(X, Y, C1, levels=[0], colors="red", linestyles="--")
    ax.contour(X, Y, C2, levels=[0], colors="blue", linestyles="--")

    for x_hist in x_histories:
        x_vals = [x[0] for x in x_hist]
        y_vals = [x[1] for x in x_hist]
        ax.plot(x_vals, y_vals, marker='o', linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_line(y_histories, ylabel, title, filename=None):
    fig, ax = plt.subplots()
    for y_vals in y_histories:
        ax.plot(y_vals, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()



data = np.load("tracking_results.npz", allow_pickle=True)
penalty = data["penalty"]
hybrid = data["hybrid"]


simple1_paths_penalty = [entry[1] for entry in penalty]
simple1_paths_hybrid = [entry[1] for entry in hybrid]

plot_contour(f1, c1, simple1_paths_penalty, "Penalty Method - Simple1", filename="simple1_penalty.png")
plot_contour(f1, c1, simple1_paths_hybrid, "Hybrid Method - Simple1", filename="simple1_hybrid.png")

plot_contour(f2, c2, [entry[1] for entry in penalty], "Penalty Method - Simple2", filename="simple2_penalty.png")
plot_contour(f2, c2, [entry[1] for entry in hybrid], "Hybrid Method - Simple2", filename="simple2_hybrid.png")

# === Line plots for simple2 ===
plot_line([entry[2] for entry in penalty], "Objective f(x)", "Penalty Method - Objective (Simple2)", filename="simple2_obj_penalty.png")
plot_line([entry[2] for entry in hybrid], "Objective f(x)", "Hybrid Method - Objective (Simple2)", filename="simple2_obj_hybrid.png")

plot_line([entry[3] for entry in penalty], "Max Constraint Violation", "Penalty Method - Violation (Simple2)", filename="simple2_violation_penalty.png")
plot_line([entry[3] for entry in hybrid], "Max Constraint Violation", "Hybrid Method - Violation (Simple2)", filename="simple2_violation_hybrid.png")
