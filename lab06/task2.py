import matplotlib.pyplot as plt
import numpy as np

from face_data import Face1, Face2, Face3, TargetFace2, edges


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


TargetFace = TargetFace2
NoisyTargetFace = TargetFace + 3 * np.random.randn(*TargetFace.shape)

face1 = Face1.ravel()
face2 = Face2.ravel()
face3 = Face3.ravel()
t = NoisyTargetFace.ravel()

F = np.stack((face1, face2, face3), axis=1)

for i in range(5):
    inds = np.random.choice(range(136), 3, replace=False)
    three_faces_sliced = F[inds]
    t_sliced = t[inds]

    x = np.linalg.solve(three_faces_sliced, t_sliced)

    a1, b1, c1 = x

    # a2, b2, c2 = np.linalg.inv(F.T @ F) @ F.T @ t
    a2, b2, c2 = np.linalg.lstsq(F, t)[0]

    Face_rnd = a1 * Face1 + b1 * Face2 + c1 * Face3
    Face_lsq = a2 * Face1 + b2 * Face2 + c2 * Face3

    plot_face(plt, NoisyTargetFace, edges, color='r')

    plot_face(plt, TargetFace, edges, color='pink')
    plot_face(plt, Face_rnd, edges, color='g')
    plot_face(plt, Face_lsq, edges, color='b')

    err_lsq = np.linalg.norm(TargetFace2.ravel() - Face_lsq.reshape(136, ))
    err_noisy = np.linalg.norm(TargetFace2.ravel() - t)
    print("err_lsq: ", err_lsq, "err_noisy: ", err_noisy)

    plt.show()
