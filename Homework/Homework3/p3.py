# import libraries
import numpy as np
def driver():
    # test functions
    f1 = lambda x: 1 + (1 + ((7-x**(1/5)) / (x ** 2))) ** 3
    f2 = lambda x: x - (((x**5) - 7) / (x ** 2))
    f3 = lambda x: x - (((x**5) - 7) / (5* (x ** 4)))
    f4 = lambda x: x - (((x**5) - 7) / 12)
    # fixed point is alpha1 = 7 ^ 1/5

    x0 = 1
    tol = 10 ** (-10)
    Nmax = 200
    
    # test f1 '''
    [xstar,ier, count] = fixedpt(f1,x0,tol,Nmax)
    print('(a) the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Number of itterations: ', count)
    print('Error message reads:',ier)

    # test f2 '''
    # [xstar,ier,count] = fixedpt(f2,x0,tol,Nmax)
    # print('(b) the approximate fixed point is:',xstar)
    # print('f2(xstar):',f2(xstar))
    # print('Number of itterations: ', count)
    # print('Error message reads:',ier)

    # test f3 '''
    [xstar,ier,count] = fixedpt(f3,x0,tol,Nmax)
    print('(c) the approximate fixed point is:',xstar)
    print('f3(xstar):',f3(xstar))
    print('Number of itterations: ', count)
    print('Error message reads:',ier)

    # test f4 '''
    [xstar,ier,count] = fixedpt(f4,x0,tol,1000)
    print('(d) the approximate fixed point is:',xstar)
    print('f4(xstar):',f4(xstar))
    print('Number of itterations: ', count)
    print('Error message reads:',ier)






# define routines
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
            #print(count)
            return [xstar,ier,count]
        
        x0 = x1
        xstar = x1

    ier = 1
    return [xstar, ier,count]
driver()