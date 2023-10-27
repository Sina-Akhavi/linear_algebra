import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2, 1],
              [2, -1, -1],
              [-1, 1, -2]])

B = np.array([[1, 2, -3],
              [3, 1, 1],
              [2, 1, 0]])

C = np.array([[1, 2, -3],
              [3, 6, -9],
              [-2, -4, 6]])

rows = 2
cols = 3

# ------------- A -------------
U2 = np.random.randn(200, 3)
row_space_A = U2 @ A

plt.subplot(rows, cols, 1, projection='3d')
plt.scatter(row_space_A[:, 0], row_space_A[:, 1], row_space_A[:, 2])
plt.title('row space of A')

U1 = np.random.randn(3, 200)
col_space_A = A @ U1

plt.subplot(rows, cols, 4, projection='3d')
plt.scatter(col_space_A[0, :], col_space_A[1, :], col_space_A[2, :])
plt.title('column space of A')
# -------------- B ----------------
U1 = np.random.randn(200, 3)
row_space_B = U1 @ B
plt.subplot(rows, cols, 2, projection='3d')
plt.scatter(row_space_B[:, 0], row_space_B[:, 1], row_space_B[:, 2])

plt.title('row space of B')

U2 = np.random.randn(3, 200)
col_space_B = B @ U2
plt.subplot(rows, cols, 5, projection='3d')
plt.scatter(col_space_B[0, :], col_space_B[1, :], col_space_B[2, :])

plt.title('column space of B')

# ------------------ C -------------------------
U1 = np.random.randn(200, 3)
row_space_C = U1 @ C
plt.subplot(rows, cols, 3, projection='3d')
plt.scatter(row_space_C[:, 0], row_space_C[:, 1], row_space_C[:, 2])
plt.title('row space of C')

U2 = np.random.randn(3, 200)
col_space_C = C @ U2
plt.subplot(rows, cols, 6, projection='3d')
plt.scatter(col_space_C[0, :], col_space_C[1, :], col_space_C[2, :])
plt.title('column space of C')
# ------------------------------- test u3 ------------------------------
# u3 = np.array([-1, 1, -2])

# base of the vectors set to the origin
# tail_x = [0, 0, 0]
# tail_y = [0, 0, 0]
# tail_z = [0, 0, 0]
#
# plt.subplot(rows, cols, 1, projection='3d')
# plt.quiver(tail_x, tail_y, tail_z, u3[0], u3[1], u3[2], color='r')


plt.show()
