import timeit

import numpy as np

m, n, p = 100, 50, 2000

A = np.random.rand(m, n, p)
A_temp = A.copy()

s = np.random.rand(p)


def f1():
    for i in range(p):
        A[:, :, i] *= s[i]


t1 = (timeit.timeit(f1, number=1000)) / 1000
t2 = (timeit.timeit(lambda: s * A_temp, number=1000)) / 1000
# Note that by increasing the 'number' of execution, the average approximately remains unchanged.

# a single line
print(t1)
print(t2)

A_temp = s.reshape(1, 1, p) * A_temp

# p is enough: the better solution

A_temp = s * A_temp
# print(A_temp)
