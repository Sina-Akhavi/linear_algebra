from spectral import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
from validate import construct_affinity_matrix


# centers = [[8, 7], [4, 2]]  # Added a third cluster center
# stds = [1, 2]  # Added a standard deviation for the third cluster

# data_1 = np.random.normal(loc=centers[0], scale=stds[0], size=(50, 2))
# data_2 = np.random.normal(loc=centers[1], scale=stds[1], size=(50, 2))
# # data_3 = np.random.normal(loc=centers[2], scale=stds[2], size=(50, 2))  # Added data for the third cluster

# # data = np.concatenate((data_1, data_2, data_3), axis=0)
# data = np.concatenate((data_1, data_2), axis=0)

# k = 2
# A = construct_affinity_matrix(data, affinity_type='rbf')
# ys = spectral_clustering(A, k)

# ------------------ plotting --------------------
# for i in range(len(ys)):
#     if ys[i] == 0:
#         plt.plot(data[i, 0], data[i, 1], "r*")
#     elif ys[i] == 1:
#         plt.plot(data[i, 0], data[i, 1], "b*")
#     else:
#         plt.plot(data[i, 0], data[i, 1], "y*")

# plt.show()



centers = [[8, 7], [4, 2], [-1, -2]]  # Added a third cluster center
stds = [1, 2, 0.5]  # Added a standard deviation for the third cluster

data_1 = np.random.normal(loc=centers[0], scale=stds[0], size=(50, 2))
data_2 = np.random.normal(loc=centers[1], scale=stds[1], size=(50, 2))
data_3 = np.random.normal(loc=centers[2], scale=stds[2], size=(50, 2))  # Added data for the third cluster

# data = np.concatenate((data_1, data_2), axis=0)
data = np.concatenate((data_1, data_2, data_3), axis=0)

A = construct_affinity_matrix(data, affinity_type='knn', k=4)

k = len(centers)
ys = spectral_clustering(A, k)

# ------------------ plotting --------------------
for i in range(len(ys)):
    if ys[i] == 0:
        plt.plot(data[i, 0], data[i, 1], "r*")
    elif ys[i] == 1:
        plt.plot(data[i, 0], data[i, 1], "b*")
    else:
        plt.plot(data[i, 0], data[i, 1], "y*")

plt.show()


