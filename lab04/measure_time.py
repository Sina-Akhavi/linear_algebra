import numpy as np
import timeit

n = 1000

A = np.random.rand(n, n)


def f():
    np.linalg.inv(A)


t1 = timeit.timeit(f, number=1)
t2 = timeit.timeit(f, number=100) / 100

# t2 is more reliable:
# twp reason for this claim:
# 1. It is the average of 100 samples
# 2. t2 is less scattered than t1 .i.e t2 remains unchanged.

print(t1)
print(t2)
