import matplotlib.pyplot as plt

from face_data import Face1, Face3, Face2, TargetFace2, edges
import numpy as np


def plot_face(plt, X, edges, color='b'):
    "plots a face"

    plt.plot(X[:, 0], X[:, 1], 'o', color=color, markersize=3)

    for i, j in edges:
        xi = X[i, 0]
        yi = X[i, 1]
        xj = X[j, 0]
        yj = X[j, 1]

        # draw a line between X[i] and X[j]
        plt.plot((xi, xj), (yi, yj), '-', color=color)
    plt.axis('square')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)


face1 = Face1.ravel()
face2 = Face2.ravel()
face3 = Face3.ravel()
t = TargetFace2.ravel()

F = np.stack((face1, face2, face3), axis=1)

NoisyTargetFace = TargetFace2 + 3 * np.random.randn(*TargetFace2.shape)
t = NoisyTargetFace.ravel()

x_est = np.linalg.inv(F.T @ F) @ F.T @ t
Face = x_est[0] * Face1 + x_est[1] * Face2 + x_est[2] * Face3

plot_face(plt, TargetFace2, edges, 'b')
plot_face(plt, Face, edges, 'yellow')

plt.show()
