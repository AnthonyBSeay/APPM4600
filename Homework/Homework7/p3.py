
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10*x)**2)

def chebyshev_interpolation(N, x_eval):
    # Chebyshev interpolation nodes
    j = np.arange(1, N + 1)
    theta = (2*j - 1) * np.pi / (2*N)
    x_nodes = np.cos(theta)
    y_nodes = f(x_nodes)

    # Barycentric weights (a common scaling factor is irrelevant)
    w = (-1.0)**(j - 1) * np.sin(theta)

    # Evaluate the interpolating polynomial
    x_eval = np.asarray(x_eval, dtype=float)
    p = np.empty_like(x_eval)

    for k, x in enumerate(x_eval):
        diff = x - x_nodes
        match = np.abs(diff) < 1e-14

        if np.any(match):
            p[k] = y_nodes[np.argmax(match)]
        else:
            terms = w / diff
            p[k] = np.sum(terms * y_nodes) / np.sum(terms)

    return x_nodes, y_nodes, p


# Fine grid for plotting
x = np.linspace(-1, 1, 2001)
y = f(x)

# Compare different numbers of Chebyshev points
N_values = [5, 10, 20, 40]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for ax, N in zip(axes, N_values):
    x_nodes, y_nodes, p = chebyshev_interpolation(N, x)

    ax.plot(x, y, "k--", linewidth=2.5, label="Original f(x)")
    ax.plot(x, p, linewidth=2, label=f"Interpolant, N={N}")
    ax.plot(x_nodes, y_nodes, "o", markersize=4,
            markerfacecolor="white", markeredgewidth=1.3,
            label="Chebyshev nodes")

    ax.set_title(f"N = {N}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("Chebyshev Interpolation",
             fontsize=15)
plt.tight_layout()
plt.show()


# Plot maximum absolute interpolation error
plt.figure(figsize=(9, 5))

for N in N_values:
    _, _, p = chebyshev_interpolation(N, x)
    error = np.abs(p - y)
    plt.semilogy(x, error, linewidth=1.8, label=f"N={N}")

plt.xlabel("x")
plt.ylabel("Absolute error |p_N(x) - f(x)|")
plt.title("Interpolation Error Using Chebyshev Points")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Report maximum errors and maximum interpolant values
for N in N_values:
    _, _, p = chebyshev_interpolation(N, x)
    print(
        f"N={N:2d} | "
        f"max error = {np.max(np.abs(p - y)):.6e} | "
        f"max p(x) = {np.max(p):.6f}"
    )
