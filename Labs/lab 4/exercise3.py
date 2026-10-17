# import libraries
import numpy as np
import scipy as sp

def driver():
    # test functions
    f = lambda x: np.exp(x**2 + 7*x -30) - 1

    f1 = lambda x: (2*x+7) * np.exp(x**2 + 7*x -30)
    
    f2 = lambda x: (4*(x**2) + 28 * x + 51) * np.exp(x**2 + 7*x -30)

    x0 = 4.5
    Nmax = 100

    [xstar,ier] = Newton(f,f1,f2,x0,Nmax)

    print('Newton:', xstar)
    print('Newton:',ier)

    a = 2
    b = 4.5
    [xstar2,ier2] = Newton_Bisection(f,f1,f2,a,b,Nmax)

    print('Modified Newton:', xstar2)
    print('Modified Newton:',ier2)
    
    
def Newton(f,f1,f2,x0,Nmax):

    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0

    g = lambda x: x - (f(x) - f1(x))

    while (count <Nmax):
        fx = f(x0)
        fpx = f1(x0)

        if (f1(x0) == 0):
            xstar = 0
            ier = 1
            return [xstar,ier]

        x1 = g(x0)

        dg_dx = lambda x: (f(x)*f2(x)) /(f1(x)**2)

        if(np.abs(dg_dx(x1)) < 1):
            xstar = x1
            ier = 0
            return [xstar,ier]

        x0 = x1

    xstar = x1
    ier = 1

    return [xstar, ier]

def Newton_Bisection(f,f1,f2,a,b,Nmax):

    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0

    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b)
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

    x0 = 0.5 * (a+b)

    g = lambda x: x - (f(x) - f1(x))

    while (count <Nmax):
        fx = f(x0)
        fpx = f1(x0)

        if (f1(x0) == 0):
            xstar = 0
            ier = 1
            return [xstar,ier]

        x1 = g(x0)

        dg_dx = lambda x: (f(x)*f2(x)) /(f1(x)**2)

        if( dg_dx(x1) < 1):
            xstar = x1
            ier = 0
            return [xstar,ier]

        x0 = x1

    xstar = x1
    ier = 1

    return [xstar, ier]

driver()