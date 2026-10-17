import numpy as np
import matplotlib.pyplot as plt

# Function to interpolate
def f(x):
    return 1 / (1 + (10 * x)**2)


# Construct the Vandermonde matrix and solve Vc = y
def interpolation_coefficients(x, y):
    N = len(x)

    # V[i, j] = x[i]**j
    V = np.vander(x, N, increasing=True)

    # Solve for the polynomial coefficients
    c = np.linalg.solve(V, y)

    return c


# Evaluate the polynomial at a point or array of points
def polynomial_evaluator(x, c):
    p = np.zeros_like(np.asarray(x, dtype=float))

    for j in range(len(c)):
        p += c[j] * np.asarray(x)**j

    return p


# Fine grid for plotting
xfine = np.linspace(-1, 1, 1001)
ffine = f(xfine)

# Try increasing values of N
for N in range(2, 21):

    # Equally spaced interpolation points
    h = 2 / (N - 1)
    x = np.array([-1 + i * h for i in range(N)])
    y = f(x)

    # Find coefficients
    c = interpolation_coefficients(x, y)

    # Evaluate polynomial on the fine grid
    pfine = polynomial_evaluator(xfine, c)

    # Print maximum and minimum polynomial values
    print(
        f"N = {N:2d}, "
        f"max(p) = {np.max(pfine):.6f}, "
        f"min(p) = {np.min(pfine):.6f}"
    )

    # Plot the polynomial, original function, and data points
    if (N == 2 or N == 5 or N == 17 or N == 20):
        plt.figure(figsize=(8, 5))
        plt.plot(xfine, ffine, label="f(x)", linewidth=2)
        plt.plot(xfine, pfine, label="p(x)", linewidth=2)
        plt.plot(x, y, 'o', label="Interpolation points")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Polynomial Interpolation, N = {N}")
        plt.grid(True)
        plt.legend()
        plt.ylim(-1.1, max(1.1, np.max(pfine) * 1.05))
        plt.show()