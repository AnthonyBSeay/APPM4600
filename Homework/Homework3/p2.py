import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def driver():
    # Plot function for part a

    Ti = 20
    Ts = -15
    alpha = 0.138e-6
    t = 60 * 24 * 60 * 60

    def f(x):
        return erf(x / (2 * np.sqrt(alpha * t))) - 3/7

    

    x = np.linspace(0, 5, 500)

    plt.plot(x, f(x))
    plt.axhline(0, linestyle='--')
    plt.xlabel("Depth x (m)")
    plt.ylabel("f(x)")
    plt.title("Water Main Freezing Depth")
    plt.grid()
    plt.show()

    # compute approximate depth using bisection

    [astar, ier,count] = bisection(f, 0, 5, 10**-13)
    print('Bisection approximate depth: ', astar)
    print('Error message:' ,ier)
    print('Itteration count: ',count)

    # compute approximate depth using Newtons method
    def f1(x):
        return (1 / np.sqrt(np.pi * alpha * t)) * \
           np.exp(-x**2 / (4 * alpha * t))

    [xstar, ier, count] = Newton(f, f1, 5, 10**-13, 100)

    print("Newton approximate depth:", xstar)
    print("Error message:", ier)
    print("Iteration count:", count)


def bisection(f,a,b,tol):
    # Inputs:
    # f,a,b - function and endpoints of initial interval
    # tol - bisection stops when interval length < tol
    # Returns:
    # astar - approximation of root
    # ier - error message
    # - ier = 1 => Failed
    # - ier = 0 == success
    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
    # verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    
    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
    # print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier,count]

def Newton(f, f1, x0, tol, Nmax):
    count = 0

    while count < Nmax:

        fx = f(x0)
        fpx = f1(x0)

        # Avoid division by zero
        if fpx == 0:
            xstar = x0
            ier = 1
            return [xstar, ier, count]

        # Newton step
        x1 = x0 - fx / fpx

        # Stopping condition
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, count + 1]

        x0 = x1
        count += 1

    xstar = x1
    ier = 1
    return [xstar, ier, count]

driver()