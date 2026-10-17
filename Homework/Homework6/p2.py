import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

def driver():

    x0_list = [np.array([1.0, 1.0]),
               np.array([1.0, -1.0]),
               np.array([0.0, 0.0])]

    Nmax = 100
    tol = 1e-10

    for x0 in x0_list:
        print('==============================================')
        print('Initial guess:', x0)


        for j in range(50):
            [xstar, ier, its] = Newton(x0, tol, Nmax)
        print('--- Newton ---')
        print(xstar)
        print('Newton: the error message reads:', ier)
        print('Newton: number of iterations is:', its)

        for j in range(20):
            [xstar, ier, its] = LazyNewton(x0, tol, Nmax)
        print('--- Lazy Newton ---')
        print(xstar)
        print('Lazy Newton: the error message reads:', ier)
        print('Lazy Newton: number of iterations is:', its)

        for j in range(20):
            [xstar, ier, its] = Broyden(x0, tol, Nmax)
        print('--- Broyden ---')
        print(xstar)
        print('Broyden: the error message reads:', ier)
        print('Broyden: number of iterations is:', its)
        print()

def evalF(x):
# vector function that you want to find the roots of

    F = np.zeros(2)

    F[0] = x[0]**2 + x[1]**2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1

    return F

def evalJ(x):
# Jacobian of the vector function you want to find the roots of

    J = np.array([[2.*x[0],       2.*x[1]],
                  [np.exp(x[0]), 1.0]])

    return J


# ier = 0: converged, ier = 1: max its reached / no convergence,
# ier = 2: Jacobian singular (cannot start / continue)

def Newton(x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    x1 = x0
    for its in range(Nmax):
       J = evalJ(x0)
       try:
           Jinv = inv(J)
       except np.linalg.LinAlgError:
           return [x0, 2, its]
       F = evalF(x0)

       x1 = x0 - Jinv.dot(F)

       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return [xstar, ier, its]

       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

def LazyNewton(x0, tol, Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    try:
        Jinv = inv(J)
    except np.linalg.LinAlgError:
        return [x0, 2, 0]
    x1 = x0
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)

       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return [xstar, ier, its]

       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

def Broyden(x0, tol, Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''

    r'''In Broyden
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''

    '''initialize with 1 newton step'''

    A0 = evalJ(x0)

    v = evalF(x0)
    try:
        A = np.linalg.inv(A0)
    except np.linalg.LinAlgError:
        return [x0, 2, 0]

    s = -A.dot(v)
    xk = x0+s
    for its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s, z)
       if p == 0 or not np.isfinite(p):
          return [xk, 1, its]
       u = np.dot(s, A)
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp, u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s) < tol):
          alpha = xk
          ier = 0
          return [alpha, ier, its]
    alpha = xk
    ier = 1
    return [alpha, ier, its]


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()