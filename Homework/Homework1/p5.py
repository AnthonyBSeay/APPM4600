import numpy as np
import matplotlib.pyplot as plt

def driver():
    # Values of delta: 10^-16, ..., 10^-1, 10^0
    delta = np.logspace(-16, 0, 100)

    # Two values of x
    x_values = [np.pi, 1e6]

    for x in x_values:
        # Original expression
        original = np.cos(x + delta) - np.cos(x)

        # Cancellation-free expression
        derived = -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

        # Plot the difference between the two
        error = np.abs(original - derived)

        plt.figure()
        plt.loglog(delta, error, 'o-')
        plt.xlabel(r'$\delta$')
        plt.ylabel('Absolute difference')
        plt.title(f'x = {x}')
        plt.grid(True, which='both')
        plt.show()

driver()