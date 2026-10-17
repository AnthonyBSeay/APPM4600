import numpy as np


def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16


def iteration(x, y, z):
    # Gradient of f
    fx = 2*x
    fy = 8*y
    fz = 8*z

    # d = f / (fx^2 + fy^2 + fz^2)
    d = f(x, y, z) / (fx**2 + fy**2 + fz**2)

    # New iterate
    x_new = x - d*fx
    y_new = y - d*fy
    z_new = z - d*fz

    return x_new, y_new, z_new


def main():

    # Initial guess
    x = 1.0
    y = 1.0
    z = 1.0

    # Iterate until convergence
    tol = 1e-12
    max_iter = 20

    # Store iterates
    points = [(x, y, z)]

    for n in range(max_iter):
        x, y, z = iteration(x, y, z)
        points.append((x, y, z))

        if abs(f(x, y, z)) < tol:
            break

    # Use the final iterate as our approximation to the exact solution
    x_star, y_star, z_star = points[-1]

    print("\nApproximate point on the ellipsoid:")
    print(f"x* = {x_star:.12f}")
    print(f"y* = {y_star:.12f}")
    print(f"z* = {z_star:.12f}")

    print("\nCheck:")
    print(f"x^2 + 4y^2 + 4z^2 = "
          f"{x_star**2 + 4*y_star**2 + 4*z_star**2:.12f}")

    # ------------------------------------------------------------
    # Iteration table
    # ------------------------------------------------------------

    print("\nIteration Table")
    print("-" * 90)

    print(f"{'n':>3} {'x_n':>15} {'y_n':>15} {'z_n':>15} "
          f"{'|f(x_n)|':>15} {'error':>15}")

    print("-" * 90)

    errors = []

    for n, (x_n, y_n, z_n) in enumerate(points):

        # Distance from final solution
        error = np.sqrt(
            (x_n - x_star)**2 +
            (y_n - y_star)**2 +
            (z_n - z_star)**2
        )

        errors.append(error)

        residual = abs(f(x_n, y_n, z_n))

        print(f"{n:3d} "
              f"{x_n:15.9f} "
              f"{y_n:15.9f} "
              f"{z_n:15.9f} "
              f"{residual:15.6e} "
              f"{error:15.6e}")

    # ------------------------------------------------------------
    # Quadratic convergence test
    # ------------------------------------------------------------

    print("\nQuadratic Convergence Test")
    print("-" * 60)

    print(f"{'n':>3} {'e_n':>18} {'e_(n+1)/e_n^2':>20}")
    print("-" * 60)

    for n in range(len(errors) - 1):

        if errors[n] > 1e-14:
            ratio = errors[n+1] / errors[n]**2

            print(f"{n:3d} "
                  f"{errors[n]:18.8e} "
                  f"{ratio:20.8e}")

    print("\nFor quadratic convergence,")
    print("e_(n+1) / e_n^2 should approach a constant.")


if __name__ == "__main__":
    main()