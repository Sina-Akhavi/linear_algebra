import timeit

import numpy as np

A = np.random.randn(100, 200)
d1 = np.random.randn(100, 1)
D1 = np.diag(d1.ravel())

t1 = timeit.timeit(lambda: d1 * A, number=100)
t2 = timeit.timeit(lambda: D1 @ A, number=100)

print('Broadcasting time: ', t1)
print('diagonal time: ', t2)

# To recap, broadcasting outperforms the utilizing diagonal matrices for matrix scaling


