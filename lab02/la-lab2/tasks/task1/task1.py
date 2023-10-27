import numpy as np
import matplotlib.pyplot as plt

n = 11
S1 = np.vstack((np.cos(np.linspace(0, np.pi, n)),
                -.7 + np.sin(np.linspace(0, np.pi, n)))).T
S2 = np.vstack((np.linspace(-1.2, 1.2, n),
                np.zeros(n))).T

plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.plot(S2[:, 0], S2[:, 1], 'bo-')
plt.plot(S1[:, 0], S1[:, 1], 'bo-')

plt.draw()
plt.pause(0.1)
#
alpha = np.linspace(0, 1, 10)
for i in alpha:
    # plt.cla()
    S3 = (1 - i) * S1 + i * S2
    plt.plot(S3[:, 0], S3[:, 1], 'ro-')

    plt.draw()
    plt.pause(0.1)

# ---------------- Vary alpha from 0 to 1.5 (affine combinations). What happens? ----------------
# alpha = np.linspace(-2, 2, 20)
# for i in alpha:
#     # plt.cla()
#     S3 = (1 - i) * S1 + i * S2
#     plt.plot(S3[:, 0], S3[:, 1], 'ro-')
#
#     plt.draw()
#     plt.pause(0.1)

plt.show()
