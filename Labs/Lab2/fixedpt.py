
# import libraries
import numpy as np
def driver():
# test functions
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....
    g1 = lambda x: np.sqrt((10 / (x+4)))
    p = 1.3652300134140976 # FIXED POINT 
    
    Nmax = 100
    x = np.zeros((Nmax,1))
    tol = 1e-6
    # test f1 '''
    #x0 = 0.0
    #[xstar, ier] = fixedpt(f1,x0,tol,Nmax,x)
    #print('the approximate fixed point is:',xstar)
    #print('f1(xstar):',f1(xstar))
    #print('Error message reads:',ier)

    x0 = 1.5
    [xstar, ier,count] = fixedpt(g1,x0,tol,Nmax,x)
    print('the approximate fixed point is:',p)
    print('g1(xstar):',g1(xstar))
    print('# of itternations:',count)
    print('Error message reads:',ier)

    print(x[0],x[1])

    print('Order of convergence ',orderOfConvergence(p, x, count))
    

# define routines
def fixedpt(f,x0,tol,Nmax,x):
    
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
        
        x1 = f(x0)
        x[count] = x1
        count = count +1
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            xnew = x[count:]
            return [xstar,ier,count]

        x0 = x1
        xstar = x1

    ier = 1
    return [xstar, ier,count]

def orderOfConvergence(p, x, k):
    num = 0
    alpha = 0
    x1 = np.log((x[0]-p) / (x[1]-p))
    x2 = np.log((x[1]-p) / (x[0]-p))

    alpha = x1/x2

    return alpha
        





driver()
