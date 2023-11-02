import numpy as np

N = 100

A = np.random.randn(N, N)

A[:, N - 1] = A[:, N - 2]

A += 0.00001 * np.random.randn(N, N)  # A became near-singular
x = np.random.randn(N)

noise = 0.001 * np.random.randn(N)


b = A @ x
b_noisy = A @ x + noise
b_noisy_emphasized = A @ x + 2 * noise

x1 = np.linalg.solve(A, b)
x2 = np.linalg.inv(A) @ b

x1_noisy = np.linalg.solve(A, b_noisy)
x2_noisy = np.linalg.inv(A) @ b_noisy

x1_noisy_emphasized = np.linalg.solve(A, b_noisy_emphasized)
x2_noisy_emphasized = np.linalg.inv(A) @ b_noisy_emphasized


# Error Comparison when noise affects the matrix
print("error1=", np.linalg.norm(x - x1))
print("error2=", np.linalg.norm(x - x2))

print("error1_with_noise=", np.linalg.norm(x - x1_noisy))
print("error2_with_noise=", np.linalg.norm(x - x2_noisy))

print("error1_with_noise_emphasized=", np.linalg.norm(x - x1_noisy_emphasized))
print("error2_with_noise_emphasized=", np.linalg.norm(x - x2_noisy_emphasized))



