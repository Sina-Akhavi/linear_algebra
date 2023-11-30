import numpy as np

x_true = np.array([3, 1.5, -1.0, 2.4, -3, -.1, 2.2, 4.1, -3.2, 1.0])
n = x_true.size  # no. of unknowns

# m = 100  # no of equations (measurements)
m = 10000  # increase m
A = np.random.randn(m, n)

# create the measurments
y_true = A @ x_true

# add noise to the measurments
sigma = 0.01
measurement_noise = sigma * np.random.randn(m)
y_noisy = y_true + measurement_noise

# we have access to the matrix "A" and noisy measurements "y_noisy".
# From these, we intend to estimate "x_true" using least squares
error_sum = 0

for i in range(100):
    x_est = np.linalg.inv(A.T @ A) @ A.T @ y_noisy
    error = np.linalg.norm(x_est - x_true)

    error_sum += error

error = error_sum / 100
print("error: ", error)


# x_est = np.linalg.solve(A.T@A, A.T @ y_noisy)
# x_est = np.linalg.lstsq(A,y_noisy)[0]

# measure the distance between the estimated unkowns "x_est"
# and the ture ones "x_true"

# print("x_true: \n", x_true, "\n\n", "x_est: \n", x_est)

# print('error=', np.linalg.norm(x_est - x_true))
