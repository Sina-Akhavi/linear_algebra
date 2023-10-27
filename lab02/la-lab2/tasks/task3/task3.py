import numpy as np

m, n = 20, 10
A = np.random.rand(m, n)
u = np.random.rand(n)

v = np.zeros(m)
for i in range(n):
    v += A[:, i] * u[i]

print(f"v=\n {v}\n\n")

# --------------- my solution ---------------
v = A @ u  # answer in a single line of code

print(f"my solution: v =:\n {v}")

# print(v)
