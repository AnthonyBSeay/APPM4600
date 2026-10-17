import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Function and derivative
# f(x) = x^6 - x - 1
# ---------------------------------------------------------

def f(x):
    return x**6 - x - 1


def df(x):
    return 6*x**5 - 1


# ---------------------------------------------------------
# Find the "exact" root alpha using a very accurate
# Newton iteration. We use this as the reference value.
# ---------------------------------------------------------

alpha = 1.1347241384015194

print("Exact root alpha =", alpha)


# ---------------------------------------------------------
# Newton's Method
# x_{k+1} = x_k - f(x_k)/f'(x_k)
# ---------------------------------------------------------

def newton(x0, tol=1e-15, max_iter=50):

    x = [x0]

    for k in range(max_iter):
        x_new = x[-1] - f(x[-1]) / df(x[-1])
        x.append(x_new)

        if abs(x_new - x[-1]) < tol:
            break

    return np.array(x)


# ---------------------------------------------------------
# Secant Method
#
# x_{k+1} =
# x_k - f(x_k)(x_k-x_{k-1})/[f(x_k)-f(x_{k-1})]
# ---------------------------------------------------------

def secant(x0, x1, tol=1e-15, max_iter=50):

    x = [x0, x1]

    for k in range(max_iter):

        x_new = (
            x[-1]
            - f(x[-1]) * (x[-1] - x[-2])
            / (f(x[-1]) - f(x[-2]))
        )

        x.append(x_new)

        if abs(x_new - x[-1]) < tol:
            break

    return np.array(x)


# ---------------------------------------------------------
# Run both methods
# ---------------------------------------------------------

newton_x = newton(2)
secant_x = secant(2, 1)


# ---------------------------------------------------------
# Calculate errors
# error_k = |x_k - alpha|
# ---------------------------------------------------------

newton_error = np.abs(newton_x - alpha)
secant_error = np.abs(secant_x - alpha)


# ---------------------------------------------------------
# Print Newton table
# ---------------------------------------------------------

print("\nNewton's Method")
print("-" * 50)
print(f"{'k':<5}{'x_k':<25}{'Error':<20}")

for k in range(len(newton_x)):
    print(f"{k:<5}{newton_x[k]:<25.16f}{newton_error[k]:.6e}")


# ---------------------------------------------------------
# Print Secant table
# ---------------------------------------------------------

print("\nSecant Method")
print("-" * 50)
print(f"{'k':<5}{'x_k':<25}{'Error':<20}")

for k in range(len(secant_x)):
    print(f"{k:<5}{secant_x[k]:<25.16f}{secant_error[k]:.6e}")


# ---------------------------------------------------------
# Part (b): Convergence plots
# ---------------------------------------------------------

# Newton
plt.figure(figsize=(7, 6))

x_newton = newton_error[:-1]
y_newton = newton_error[1:]

# Remove machine precision points
mask = (x_newton > 1e-14) & (y_newton > 1e-14)

plt.loglog(x_newton[mask], y_newton[mask], 'o-', label="Newton")

# Reference slope-2 line
x_ref = np.array([1e-1, 1e-2, 1e-3])
y_ref = x_ref**2

plt.loglog(x_ref, y_ref, '--', label="Slope = 2")

plt.xlabel(r"$|x_k-\alpha|$")
plt.ylabel(r"$|x_{k+1}-\alpha|$")
plt.title("Newton's Method")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------
# Secant
# ---------------------------------------------------------

plt.figure(figsize=(7, 6))

x_secant = secant_error[:-1]
y_secant = secant_error[1:]

mask = (x_secant > 1e-14) & (y_secant > 1e-14)

plt.loglog(x_secant[mask], y_secant[mask], 's-', label="Secant")

# Reference slope-1.618 line
p = (1 + np.sqrt(5)) / 2

x_ref = np.array([1e-1, 1e-2, 1e-3])
y_ref = x_ref**p

plt.loglog(x_ref, y_ref, '--', label=f"Slope = {p:.3f}")

plt.xlabel(r"$|x_k-\alpha|$")
plt.ylabel(r"$|x_{k+1}-\alpha|$")
plt.title("Secant Method")
plt.legend()
plt.grid(True, which="both")
plt.show()

# ---------------------------------------------------------
# Estimate slopes using linear regression on log errors
#
# log(e_{k+1}) = p log(e_k) + C
#
# slope = p = order of convergence
# ---------------------------------------------------------

# Remove errors that are zero or too close to machine precision
def estimate_order(errors):

    x = errors[:-1]
    y = errors[1:]

    mask = (x > 1e-14) & (y > 1e-14)

    slope, intercept = np.polyfit(
        np.log(x[mask]),
        np.log(y[mask]),
        1
    )

    return slope


newton_order = estimate_order(newton_error)
secant_order = estimate_order(secant_error)

print("\nEstimated convergence orders:")
print(f"Newton: {newton_order:.4f}")
print(f"Secant: {secant_order:.4f}")