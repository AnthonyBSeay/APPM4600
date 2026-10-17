
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# Original function
def f(x):
    return 1 / (1 + (10*x)**2)


# Equally spaced nodes and barycentric interpolation
def interpolate(N, x):
    xnodes = np.linspace(-1, 1, N)
    ynodes = f(xnodes)

    p = BarycentricInterpolator(xnodes, ynodes)
    return xnodes, ynodes, p(x)


# Fine grid
xfine = np.linspace(-1, 1, 4001)
ffine = f(xfine)

# Consistent colors
colors = {
    10: "tab:green",
    20: "tab:orange",
    50: "tab:purple",
    100: "tab:red"
}


# =====================================================
# FIGURE 1: Full interval -- show Runge's phenomenon
# =====================================================
N = 20
xnodes, ynodes, p = interpolate(N, xfine)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(xfine, ffine, color="black", linewidth=2.5,
        label=r"Original $f(x)$")
ax.plot(xfine, p, color="tab:orange", linewidth=2,
        label=fr"Barycentric polynomial ($N={N}$)")
ax.scatter(xnodes, ynodes, color="tab:blue", s=42,
           edgecolor="white", linewidth=0.6, zorder=3,
           label="Interpolation nodes")

ax.set_xlim(-1, 1)
ax.set_ylim(-5, 40)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
ax.set_title("Runge's Phenomenon: Full Interval", fontsize=15)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()


# =====================================================
# FIGURE 2: Endpoint zoom -- focus on oscillations
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, interval, title in [
    (axes[0], (-1, -0.65), "Left endpoint"),
    (axes[1], (0.65, 1), "Right endpoint")
]:
    ax.plot(xfine, ffine, color="black", linewidth=2.5,
            label=r"$f(x)$")

    for N in [10, 20]:
        xnodes, ynodes, p = interpolate(N, xfine)
        ax.plot(xfine, p, color=colors[N], linewidth=1.8,
                label=fr"$N={N}$")

        mask = (xnodes >= interval[0]) & (xnodes <= interval[1])
        ax.scatter(xnodes[mask], ynodes[mask],
                   color=colors[N], s=25, zorder=3)

    ax.set_xlim(*interval)
    ax.set_ylim(-5, 40)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("$x$")
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_ylabel("$p(x)$")
fig.suptitle("Endpoint Oscillations with Equally Spaced Nodes",
             fontsize=15)
fig.tight_layout()
plt.show()


# =====================================================
# FIGURE 3: Center zoom -- show improved local behavior
# =====================================================
xcenter = np.linspace(-0.2, 0.2, 1501)
fcenter = f(xcenter)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(xcenter, fcenter, color="black", linewidth=3,
        linestyle="--", label=r"Original $f(x)$", zorder=5)

for N in [10, 20, 50]:
    xnodes, ynodes, pcenter = interpolate(N, xcenter)

    ax.plot(xcenter, pcenter, color=colors.get(N, "tab:blue"),
            linewidth=1.8, label=fr"$N={N}$")

ax.set_xlim(-0.1, 0.1)
ax.set_ylim(0.1, 1.08)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
ax.set_title("Near the Center: Accuracy Improves as N Increases",
             fontsize=15)
ax.grid(alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
plt.show()


# =====================================================
# Numerical comparison near the center
# =====================================================
print("Maximum absolute error on [-0.2, 0.2]")
for N in [10, 20, 50, 100]:
    _, _, pcenter = interpolate(N, xcenter)
    error = np.max(np.abs(pcenter - fcenter))
    print(f"N = {N:3d}: error = {error:.6e}")
