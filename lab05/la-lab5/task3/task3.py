import numpy as np

N = 100

A = np.random.randn(N, N)
A[:, N - 1] = A[:, N - 2]
# A is singular due to the equality of Nth and (N + 1)
A += 0.0000001 * np.random.randn(N, N)

A_non_singular = np.random.randn(N, N)

x = np.random.randn(N)
b = A @ x

b_non_singular = A_non_singular @ x

# solve the linear equation for near-singular case
x1 = np.linalg.solve(A, b)

# solve the linear equation for near-singular case
x1_non_singular = np.linalg.solve(A_non_singular, b_non_singular)

# error comparison
print('the error for the singular case: ', np.linalg.norm(x1 - x))  # the magnitude of the error is 10^-9
print('the error for the non-singular case: ', np.linalg.norm(x1_non_singular - x))  # the magnitude of the error is
# 10^-13

# By decreasing the noise, the  error of the near singular case rises. Conversely, The error of the non-singular case
# falls.


