import numpy as np

a = np.array([[1, 2, 7, 4],
              [-1, 0, 2, 1],
              [3, 2, 1, 0]])

b = np.array([[10], [20], [-10], [30]])

c1 = a @ b
b1 = b.flatten()

c2 = a @ b1

print(f"{c1}\n\n {c2}")

