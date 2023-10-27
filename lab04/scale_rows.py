import timeit

import numpy as np

d1 = np.array([2, 3, 4]).reshape((3, 1))
# d1 = np.array([2, 3, 4])  # error. reshaping is necessary

print('d1=\n', d1)
A = np.array([[1, 1, 1, 1],
              [1, 2, 2, 2],
              [1, 2, 3, 4]])

# print('d1=\n', d1)

#
# print('d1.shape=\n', d1.shape)
# print('A.shape=\n', A.shape)
#
print('d1 * A=\n', d1 * A)

# Write an equivalent code to scale columns of a matrix with numbers
# [10,20,30,40]

# d2 = np.array([10, 20, 30, 40])

# print('d2=\n', d2)
# print('d2 * A=\n', d2 * A)

# Measure the execution time of d1*A and D1@A using timeit Which one is
# faster? Why?

# D1 = np.diag([2, 3, 4])
# print('D1=\n', D1)

# print('A=\n', A)

# execution_times = 100

# t1 = timeit.timeit(lambda: d1 * A, number=execution_times) / execution_times
# t2 = timeit.timeit(lambda: D1 @ A, number=execution_times) / execution_times

# print('D1 @ A= \n', D1 @ A)
# print('d1 * A= \n', d1 * A)

# print('execution time of d1 * A: ', t1)
# print('execution time of D1 @ A: ', t2)

# D1 @ A runs faster than d1 * A. (Why?)
# But I think that broadcasting should be more efficient.

# the second way to compare the efficiency of the two scaling approaches:

# import time

print('----------------------- second approach -----------------------')
# Create a large matrix
matrix = np.random.rand(1000, 1000)

# Create a vector of scaling factors
scaling_factors = np.random.rand(1000)

# Time the broadcasting method
# start = time.perf_counter()
# scaled_matrix_broadcast = matrix * scaling_factors
# end = time.perf_counter()
# broadcast_time = end - start

# Time the diagonal matrix method
# start = time.perf_counter()
# scaled_matrix_diagonal = matrix @ np.diag(scaling_factors)
# end = time.perf_counter()
# diagonal_time = end - start

# Print the performance results
# print("Broadcasting time:", broadcast_time)
# print("Diagonal matrix time:", diagonal_time)

# ----------------------- Nasihatkon approach --------------------------

t1 = timeit.timeit(lambda: matrix * scaling_factors, number=100)
t2 = timeit.timeit(lambda: matrix @ np.diag(scaling_factors), number=100)

print('\n\nbroadcasting time: ', t1)
print('diagonal matrix time: ', t2)

# conclusion: broadcasting outperforms the diagonal matrices for scaling the matrices. In contrast, if the matrix is
# small, the diagonal performs more efficiently.
