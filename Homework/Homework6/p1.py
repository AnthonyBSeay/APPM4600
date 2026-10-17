import numpy as np
import time
from numpy.linalg import norm
from numpy.linalg import inv
import warnings
warnings.filterwarnings("ignore")   # silence overflow warnings when iteration diverges

# G(x) = x - A*F(x)
A = np.array([[0.016, -0.17],
            [0.52,  -0.26]])

def driver():

    Nmax = 100
    tol = 1e-4

    x0_list = [np.array([-0.5, 0.25]),
               np.array([-1.5, 1.25]),
               np.array([-3.0, 3.0])]

    for x0 in x0_list:
        t = time.time()
        for j in range(50):
            [xstar, ier, its] = fixedSys(x0, tol, Nmax)
        elapsed = time.time() - t

        print('Initial guess:', x0)
        print('  approx root:', xstar)
        print('  Fixed pt: the error message reads:', ier)
        #print('  Fixed pt: took this many seconds:', elapsed / 50)
        print('  Fixed pt: number of iterations is:', its)
        print()

    print('================= NEWTONS METHOD ===================')

    for x0 in x0_list:
        t = time.time()
        for j in range(50):
            [xstar, ier, its] = Newton(x0, tol, Nmax)
        elapsed = time.time() - t
 
        print('Initial guess:', x0)
        print('  approx root:', xstar)
        print('  Newton: the error message reads:', ier)
        print('  Newton: number of iterations is:', its)
        print('  ||F(xstar)|| =', norm(evalF(xstar)))
        print()


def evalF(x):
    ''' the nonlinear system F(x,y) = 0 '''
    F = np.zeros(2)
    F[0] = 3 * x[0]**2 + 4 * x[1]**2 - 1
    F[1] = x[1]**3 - 8 * x[0]**3 - 1
    return F



def evalG(x):
    ''' fixed point map G(x) = x - A F(x) '''
    return x - A @ evalF(x)

def evalJ(x):
 
    J = np.array([[6.*x[0],        8.*x[1]],
                  [-24.*x[0]**2,   3.*x[1]**2]])
 
    return J


def fixedSys(x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):

        x1 = evalG(x0)

        # stop early if the iteration has blown up
        if (not np.all(np.isfinite(x1))) or norm(x1) > 1e100:
            xstar = x1
            ier = 1
            return [xstar, ier, its]

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

def Newton(x0, tol, Nmax):
 
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
 
    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
 
       x1 = x0 - Jinv.dot(F)
 
       if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return [xstar, ier, its]
 
       x0 = x1
 
    xstar = x1
    ier = 1
    return [xstar, ier, its]
 
driver()
