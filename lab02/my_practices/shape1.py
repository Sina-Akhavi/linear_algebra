import numpy as np
import matplotlib.pyplot as plt

n = 11
S1 = np.vstack((-np.cos(np.linspace(0, np.pi, n)),
                -.7 + np.sin(np.linspace(0, np.pi, n)))).T
S2 = np.vstack((np.linspace(-1.2, 1.2, n),
                np.zeros(n))).T

print(S1.shape)
print(S2.shape)
plt.plot(S1[:, 0], S1[:, 1], 'bo-')
plt.plot(S2[:, 0], S2[:, 1], 'ro-')
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
# -------------------------------------------------
a, b = 0.5, 0.5

S3 = a * S1 + b * S2
plt.plot(S3[:, 0], S3[:, 1], 'go-')

plt.show()

#
# print( (-np.cos(np.linspace(0, np.pi, n)), -.7 + np.sin(np.linspace(0, np.pi, n))) )
#
# arr1 = np.vstack(([1, 2, 3], [2, 4, 7]))
# print(arr1)
