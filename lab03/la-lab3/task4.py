import numpy as np
import matplotlib.pyplot as plt
from face_data import Face1, Face2, Face3, edges


def plot_face(plt, X, edges, color='b'):
    "plots a face"
    plt.plot(X[:, 0], X[:, 1], 'o', color=color)

    for i, j in edges:
        xi, yi = X[i]
        xj, yj = X[j]

        plt.plot((xi, xj), (yi, yj), '-', color=color)

        plt.axis('square')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)


# --------------------- A ---------------------

thethas = np.linspace(0, 2 * np.pi, 20)

for thetha in thethas:
    plt.cla()
    A = np.array([[np.cos(thetha), np.sin(thetha)]
                     , [-np.sin(thetha), np.cos(thetha)]])

    rotated_face = Face1 @ A
    plot_face(plt, rotated_face, edges)
    plt.draw()
    plt.pause(0.2)

# --------------------- B ---------------------
# a = np.linspace(3 / 4, 4/3, 20)
#
# for single_a in a:
#     plt.cla()
#     A = single_a * np.eye(2)
#     scaled_face = Face1 @ A
#     plot_face(plt, scaled_face, edges, color='g')
#     plt.draw()
#     plt.pause(0.5)

# ------------------- C -------------------
# alphas = np.linspace(3 / 4, 4 / 3, 20)
# for alpha in alphas:
#     plt.cla()
#     A = np.array([[alpha, 0],
#                   [0, 1/alpha]])
#
#     non_uniformed_scaled_face = Face1 @ A
#     plot_face(plt, non_uniformed_scaled_face, edges, color='y')
#
#     plt.draw()
#     plt.pause(0.5)

# ------------------- D -------------------
# s = np.linspace(-0.7, 0.7, 20)
#
# for single_s in s:
#     plt.cla()
#     A = np.array([[1, single_s],
#                   [0, 1]])
#
#     mapped_face = Face1 @ A
#     plot_face(plt, mapped_face, edges, color='orange')
#
#     plt.draw()
#     plt.pause(0.5)
