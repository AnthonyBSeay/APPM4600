# import libraries
import numpy as np
import matplotlib.pyplot as plt


def driver():
    # define x
    x = np.arange(1.920, 2.080 + 0.001, 0.001)

    p1 = lambda x: (x**9 - 18*x**8 + 144*x**7 - 672*x**6
                     + 2016*x**5 - 4032*x**4 + 5376*x**3
                     - 4608*x**2 + 2304*x - 512)

    p2 = lambda x: (x - 2)**9

    y1 = p1(x)
    y2 = p2(x)

    plt.plot(x, y1)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("p(x) Evaluated Using Coefficients")
    plt.grid()
    plt.show()

    plt.plot(x, y2)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("p(x) Evaluated Using (x - 2)^9")
    plt.grid()
    plt.show()

driver()


