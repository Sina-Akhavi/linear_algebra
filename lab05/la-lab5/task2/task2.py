import numpy as np

N = 100

A = np.random.randn(N, N)
x = np.random.randn(N)

A[:, N - 1] = A[:, N - 2]
# A is singular due to the equality of Nth and (N + 1)
b = A @ x

x2 = np.linalg.solve(A, b)

print(np.linalg.norm(x - x2))
x1 = np.linalg.inv(A) @ b  # in most of the time, it gives the error 'singular matrix'. However, I have faced some
# cases that the code runs completely and I don't know why.
