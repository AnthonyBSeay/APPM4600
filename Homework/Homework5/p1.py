import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    # Define the functions
    def f(x, y):
        return 3*x**2 - y**2

    def g(x, y):
        return 3*x*y**2 - x**3 - 1

    # Starting point
    # x0 = 1.0
    # y0 = 1.0

    # (x,y) = a(f,g,x0,y0,50)

    # print("\nFinal approximation:")
    # print("x =", x)
    # print("y =", y)
    # print("f(x,y) =", f(x, y))
    # print("g(x,y) =", g(x, y))


    # Newton method ------------------------------------------

    # Initial guess
    x0 = np.array([1.0, 1.0])

    Nmax = 100
    tol = 1e-10

    xstar, ier, its = Newton(x0, tol, Nmax)

    print("\nFinal approximation:")
    print("x =", xstar[0])
    print("y =", xstar[1])

    # Evaluate functions at solution
    F = evalF(xstar)

    print("f(x,y) =", F[0])
    print("g(x,y) =", F[1])

    print("Number of iterations =", its)
    print("Error =", norm(F))





def a(f,g,x0,y0,Nmax):
    x = x0
    y = y0
    
    # Iteration matrix
    A = np.array([
        [1/6, 1/18],
        [0,   1/6]
    ])

    for n in range(Nmax + 1):
        # Calculate residuals
        fn = f(x, y)
        gn = g(x, y)

        print(f"{n:3d} {x:15.10f} {y:15.10f} {abs(fn):15.6e} {abs(gn):15.6e}")

        # Apply iteration
        F = np.array([fn, gn])
        xy = np.array([x, y])

        xy_new = xy - A @ F

        x, y = xy_new

    return (x,y)

def evalF(x):
    """
    Evaluate F(x,y):
        f(x,y) = 3x^2 - y^2
        g(x,y) = 3xy^2 - x^3 - 1
    """

    F = np.zeros(2)

    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1

    return F


def evalJ(x):
    """
    Jacobian matrix:
    
        [ 6x              -2y          ]
    J = [                               ]
        [ 3y^2 - 3x^2      6xy         ]
    """

    J = np.array([
        [6*x[0], -2*x[1]],
        [3*x[1]**2 - 3*x[0]**2, 6*x[0]*x[1]]
    ])

    return J


def Newton(x0, tol, Nmax):
    """
    Newton's method for a 2D nonlinear system.

    Inputs:
        x0  = initial guess
        tol = tolerance
        Nmax = maximum number of iterations

    Outputs:
        xstar = approximate solution
        ier   = error flag
        its   = number of iterations
    """

    x = x0.copy()

    print(f"{'n':>3} {'x_n':>15} {'y_n':>15} "
          f"{'|f|':>15} {'|g|':>15}")

    for its in range(Nmax):

        # Evaluate F and Jacobian
        F = evalF(x)
        J = evalJ(x)

        # Newton step
        Jinv = inv(J)
        xnew = x - Jinv.dot(F)

        # Calculate residuals at current point
        print(f"{its:3d} {x[0]:15.10f} {x[1]:15.10f} "
              f"{abs(F[0]):15.6e} {abs(F[1]):15.6e}")

        # Check convergence
        if norm(xnew - x) < tol:
            xstar = xnew
            ier = 0
            return xstar, ier, its + 1

        x = xnew

    xstar = x
    ier = 1

    return xstar, ier, Nmax



driver()

