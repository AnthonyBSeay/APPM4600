import numpy as np

A = 0.5 * np.array([
    [1, 1],
    [1 + 10**(-10), 1 - 10**(-10)]
])

A_inv =  np.array([
    [1 - 10**(10), 10**(10)],
    [1 + 10**(10), -10**(10)]
])

norm_A = np.linalg.norm(A, 2)
norm_A_inv = np.linalg.norm(A_inv, 2)

condition_number = norm_A * norm_A_inv

np.set_printoptions(precision=12)

print("A =", A)
print("A inverse =", A_inv)
print("||A||_2 =", norm_A)
print("||A^-1||_2 =", norm_A_inv)
print("condition number =", condition_number)