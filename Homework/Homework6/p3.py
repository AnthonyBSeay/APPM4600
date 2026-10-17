"""
Problem 3: solve the nonlinear system
    f1 = x + cos(xyz) - 1                          = 0
    f2 = (1-x)^(1/4) + y + 0.05 z^2 - 0.15 z - 1  = 0
    f3 = -x^2 - 0.1 y^2 + 0.01 y + z - 1           = 0
with (1) Newton, (2) Steepest descent, (3) Steepest descent (tol 5e-2) -> Newton.
Error message: 0 = converged within tolerance, 1 = failed (max iterations / breakdown).
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TOL = 1e-6
GUESSES = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [-1.0, 1.0, -1.0], [0.9, 2.0, 3.0]]

def F(v):
    x, y, z = v
    return np.array([
        x + np.cos(x*y*z) - 1,
        (1 - x)**0.25 + y + 0.05*z**2 - 0.15*z - 1,
        -x**2 - 0.1*y**2 + 0.01*y + z - 1])

def J(v):
    x, y, z = v
    s = np.sin(x*y*z)
    return np.array([
        [1 - y*z*s,            -x*z*s,           -x*y*s],
        [-0.25*(1 - x)**-0.75,  1.0,              0.1*z - 0.15],
        [-2*x,                 -0.2*y + 0.01,     1.0]])

def g(v):      return F(v) @ F(v)
def grad_g(v): return 2 * J(v).T @ F(v)

def newton(x0, tol=TOL, maxit=100):
    """returns (x, error_flag, iterations)"""
    x = np.array(x0, float)
    for k in range(1, maxit + 1):
        try:
            dx = np.linalg.solve(J(x), -F(x))
        except np.linalg.LinAlgError:
            return np.full(3, np.nan), 1, k
        x = x + dx
        if not np.all(np.isfinite(x)):
            return np.full(3, np.nan), 1, k
        if np.linalg.norm(dx, np.inf) < tol:
            return x, 0, k
    return x, 1, maxit

def steepest_descent(x0, tol=TOL, maxit=100000):
    """returns (x, error_flag, iterations). Flag is 1 if the method stalls at a
    point that is not a root (||F|| > 1e-3), e.g. a local minimum of g."""
    x, err, k = _sd(x0, tol, maxit)
    if err == 0 and not (np.linalg.norm(F(x)) < 1e-3):
        err = 1
    return x, err, k

def _sd(x0, tol, maxit):
    x = np.array(x0, float)
    g1 = g(x)
    if not np.isfinite(g1):
        return np.full(3, np.nan), 1, 0
    for k in range(1, maxit + 1):
        z = grad_g(x)
        z0 = np.linalg.norm(z)
        if z0 == 0:
            return x, 0, k
        z = z / z0
        a3 = 1.0
        g3 = g(x - a3*z)
        while not (g3 < g1):                 # backtrack (also handles nan)
            a3 /= 2; g3 = g(x - a3*z)
            if a3 < tol/2:
                return x, 0, k               # no further improvement possible
        a2 = a3/2; g2 = g(x - a2*z)
        h1 = (g2 - g1)/a2
        h2 = (g3 - g2)/(a3 - a2)
        h3 = (h2 - h1)/a3
        a0 = 0.5*(a2 - h1/h3)
        g0 = g(x - a0*z)
        a, gm = (a0, g0) if g0 < g3 else (a3, g3)
        x = x - a*z
        if abs(gm - g1) < tol:
            return x, 0, k
        g1 = gm
    return x, 1, maxit

def sd_then_newton(x0):
    xc, e1, k1 = _sd(x0, 5e-2, 100000)
    if e1 != 0 or not np.all(np.isfinite(xc)):
        return xc, 1, k1
    xn, e2, k2 = newton(xc)
    return xn, e2, k1 + k2

def show(name, result):
    x, err, its = result
    print(f"--- {name} ---")
    print(x)
    print(f"{name}: the error message reads: {err}")
    print(f"{name}: number of iterations is: {its}")

if __name__ == "__main__":
    np.set_printoptions(precision=8)
    for x0 in GUESSES:
        print("==============================================")
        print("Initial guess:", np.array(x0))
        show("Newton", newton(x0))
        show("Steepest Descent", steepest_descent(x0))
        show("SD(5e-2) + Newton", sd_then_newton(x0))
        print()
    print("==============================================")