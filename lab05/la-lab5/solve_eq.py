import numpy as np
import timeit

N = 100
# N = 1000
# solve method is still more accurate
p = 1000

A = np.random.randn(N, N)
# x = np.random.randn(N)
x = np.random.randn(N, p)
b = A @ x

# solve for x given A and b
x1 = np.linalg.solve(A, b)
x2 = np.linalg.inv(A) @ b
# using the solve method performs more accurately

print("error1=", np.linalg.norm(x - x1))  # error of solve method
print("error2=", np.linalg.norm(x - x2))  # # error of inv method

print("elapsed1=", timeit.timeit(lambda: np.linalg.solve(A, b), number=100))
print("elapsed2=", timeit.timeit(lambda: np.linalg.inv(A) @ b, number=100))

# Which method is more accurate? Using np.linalg.solve or using the
# inverse? solve method

# Set N to a larger number and look at the results.
# result: the solve method is more accurate and faster

# Set the true x to a matrix x = np.random.randn(N,P) with P=100, so that
# b becomes a matrix of the same size. Which method is faster? Choose a
# larger P. What happens?
# result: inv runs faster when p = 100. By increasing the p, the inv method performs much aster than the solve method.
