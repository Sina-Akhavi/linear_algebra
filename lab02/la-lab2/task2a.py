import matplotlib.pyplot as plt
import numpy as np

from face_data import Face1, Face2, Face3, edges


def plot_face(plt, X, edges, color='b'):
    "plots a face"

    plt.plot(X[:, 0], X[:, 1], 'o', color=color)

    for edge in edges:
        xi = X[edge[0], 0]
        yi = X[edge[0], 1]

        xj = X[edge[1], 0]
        yj = X[edge[1], 1]

        # draw a line between X[i] and X[j]
        plt.plot((xi, xj), (yi, yj), '-', color=color)

    plt.axis('square')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)


# -------------- morph one face to another face --------------
plot_face(plt, Face1, edges, color='b')
plot_face(plt, Face2, edges, color='r')
# plot_face(plt, Face3, edges, color='y')
#
alpha = np.linspace(0, 1, 10)
alpha = alpha[::-1]
#
plt.draw()
plt.pause(2)

for i in alpha:
    plt.cla()
    Face = i * Face1 + (1 - i) * Face2
    plot_face(plt, Face, edges, 'g')
    plt.draw()
    plt.pause(0.2)

plt.show()
