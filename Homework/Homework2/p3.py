import math
import numpy as np

def driver():
    x = 9.999999995e-10

    # y = math.e**x
    # print(y-1)

    true_value = np.expm1(x)
    approx = x + x**2 / 2

    print(f"true value       = {true_value:.20e}")
    print(f"approximation    = {approx:.20e}")
    print(f"relative error   = {abs(approx-true_value)/abs(true_value):.20e}")


driver()