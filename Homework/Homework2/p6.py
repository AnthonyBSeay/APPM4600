import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: x - 4*np.sin(2*x) - 3

    # Choose a large interval
    x = np.linspace(-10, 10, 10000)
    y = f(x)

    # Plot
    plt.plot(x, y, label=r"$f(x)=x-4\sin(2x)-3$")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(r"$f(x)=x-4\sin(2x)-3$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # fixedpt itteration for f(x)
    f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4

    initial_guesses = np.linspace(-1,8, 15)
    for x0 in initial_guesses:
        root, ier = fixedpt(f, x0, 1e-12, 10000)
        print(f"x0 = {x0:.2f}, root = {root:.10f}, ier = {ier}")


def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]

driver()