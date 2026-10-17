import numpy as np

# ============================================================
# Problem
# f(x) = e^(3x) - 27x^6 + 27x^4 e^x - 9x^2 e^(2x)
#
# f(x) = (e^x - 3x^2)^3
#
# Root in (3,5):
# alpha ≈ 3.7330790286
# ============================================================

alpha = 3.7330790286
x0 = 4.0
tol = 1e-14
max_iter = 100


# ============================================================
# f(x), f'(x), f''(x)
# ============================================================

def f(x):
    return (
        np.exp(3*x)
        - 27*x**6
        + 27*x**4*np.exp(x)
        - 9*x**2*np.exp(2*x)
    )


def fp(x):
    return (
        3*np.exp(3*x)
        - 162*x**5
        + 27*np.exp(x)*(4*x**3 + x**4)
        - 18*x*np.exp(2*x)
        - 18*x**2*np.exp(2*x)
    )


def fpp(x):
    return (
        9*np.exp(3*x)
        - 810*x**4
        + 27*np.exp(x)*(12*x**2 + 8*x**3 + x**4)
        - 18*np.exp(2*x)*(1 + 4*x + 2*x**2)
    )


# ============================================================
# Calculate observed order
#
# p_k =
# log(e_{k+1}/e_k) / log(e_k/e_{k-1})
# ============================================================

def calculate_order(e_prev, e_current, e_next):

    if e_prev <= 0 or e_current <= 0 or e_next <= 0:
        return np.nan

    denominator = np.log(e_current / e_prev)

    if abs(denominator) < 1e-15:
        return np.nan

    return np.log(e_next / e_current) / denominator


# ============================================================
# (i) Ordinary Newton
# ============================================================

def ordinary_newton():

    x = x0
    results = []

    for k in range(max_iter):

        error = abs(x - alpha)

        results.append([k, x, f(x), error, np.nan])

        if error < tol:
            break

        x_new = x - f(x) / fp(x)

        # Calculate observed p once we have enough errors
        if len(results) >= 3:

            e_prev = results[-3][3]
            e_current = results[-2][3]
            e_next = error

            results[-2][4] = calculate_order(
                e_prev,
                e_current,
                e_next
            )

        x = x_new

    return results


# ============================================================
# (ii) Modified Newton
#
# x_{k+1} = x_k - m f/f'
#
# m = 3
# ============================================================

def modified_newton():

    x = x0
    m = 3
    results = []

    for k in range(max_iter):

        error = abs(x - alpha)

        results.append([k, x, f(x), error, np.nan])

        if error < tol:
            break

        x_new = x - m * f(x) / fp(x)

        if len(results) >= 3:

            e_prev = results[-3][3]
            e_current = results[-2][3]
            e_next = error

            results[-2][4] = calculate_order(
                e_prev,
                e_current,
                e_next
            )

        x = x_new

    return results


# ============================================================
# (iii) Newton applied to
#
# g(x) = f(x)/f'(x)
#
# g'(x) =
# 1 - f(x)f''(x)/(f'(x))^2
# ============================================================

def newton_on_g():

    x = x0
    results = []

    for k in range(max_iter):

        error = abs(x - alpha)

        results.append([k, x, f(x), error, np.nan])

        if error < tol:
            break

        g = f(x) / fp(x)

        gp = 1 - (f(x) * fpp(x)) / (fp(x)**2)

        x_new = x - g / gp

        if len(results) >= 3:

            e_prev = results[-3][3]
            e_current = results[-2][3]
            e_next = error

            results[-2][4] = calculate_order(
                e_prev,
                e_current,
                e_next
            )

        x = x_new

    return results


# ============================================================
# Print results
# ============================================================

def print_results(name, results):

    print("\n" + "="*80)
    print(name)
    print("="*80)

    print(
        f"{'k':>3} "
        f"{'x_k':>20} "
        f"{'f(x_k)':>20} "
        f"{'error':>20} "
        f"{'p':>10}"
    )

    print("-"*80)

    for row in results:

        k, x, fx, error, p = row

        if np.isnan(p):
            p_string = ""
        else:
            p_string = f"{p:.6f}"

        print(
            f"{k:3d} "
            f"{x:20.12f} "
            f"{fx:20.8e} "
            f"{error:20.8e} "
            f"{p_string:>10}"
        )


# ============================================================
# Run all methods
# ============================================================

ordinary = ordinary_newton()
modified = modified_newton()
g_method = newton_on_g()


print_results(
    "METHOD 1: ORDINARY NEWTON",
    ordinary
)

print_results(
    "METHOD 2: MODIFIED NEWTON (m = 3)",
    modified
)

print_results(
    "METHOD 3: NEWTON ON g(x) = f(x)/f'(x)",
    g_method
)


# ============================================================
# Final comparison
# ============================================================

print("\n")
print("="*80)
print("FINAL COMPARISON")
print("="*80)

print(
    f"{'Method':<35}"
    f"{'Iterations':>12}"
    f"{'Final x':>20}"
    f"{'Final error':>20}"
)

print("-"*90)

methods = [
    ("Ordinary Newton", ordinary),
    ("Modified Newton (m=3)", modified),
    ("Newton on g=f/f'", g_method)
]

for name, results in methods:

    final = results[-1]

    iterations = final[0]
    final_x = final[1]
    final_error = final[3]

    print(
        f"{name:<35}"
        f"{iterations:>12}"
        f"{final_x:>20.12f}"
        f"{final_error:>20.8e}"
    )


print("\n")
print("="*80)
print("THEORETICAL ORDERS")
print("="*80)

print("Ordinary Newton:        p = 1 (linear)")
print("Modified Newton:       p = 2 (quadratic)")
print("Newton on g=f/f':      p = 2 (quadratic)")