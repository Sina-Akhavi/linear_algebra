import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score

def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):


    # TODO: Compute pairwise distances
    distances = np.linalg.norm(data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2)
    
    if affinity_type == 'knn':
        n = data.shape[0]
        # print("n=",n)
        # TODO: Find k nearest neighbors for each point
        knn_indices = np.argsort(distances, axis=-1)[:, :k]
        # TODO: Construct symmetric affinity matrix
        # print("m=",m)
        affinity_matrix = np.zeros((n, n), dtype=int)
        affinity_matrix[np.arange(n)[:, None], knn_indices] = 1.0 # this may not be a symmetric matrix
        # Symmetrize the matrix 
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        # TODO: Return affinity matrix
        return affinity_matrix
 
    elif affinity_type == 'rbf':
        # TODO: Apply RBF kernel
        gamma = 1.0 / (2 * sigma**2)
        A = np.exp(-gamma * distances**2)
        # TODO: Return affinity matrix
        return A
    else:
        raise Exception("invalid affinity matrix type")


data1 = np.array([[1, 2], [3, 4], [5, 6]])
data2 = np.array([[3, 4], [6, 1], [7, 1]])


distances = np.linalg.norm(data1[:, np.newaxis, :] -  - data2[np.newaxis, :, :], axis=2)

# distances = np.linalg.norm(data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2)
# n = data.shape[0]

print("distances: \n", distances)

# k = 3
# knn_indices = np.argsort(distances, axis=-1)[:, :k]
# print("knn_indices: \n", knn_indices)

# TODO: Construct symmetric affinity matrix

# affinity_matrix = np.zeros((n, n), dtype=int)
# affinity_matrix[np.arange(n)[:, None], knn_indices] = 1.0 # this may not be a symmetric matrix

# print("affinity_matrix before symmetrizatiosn: \n", affinity_matrix)

# affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

# print("affinity_matrix after symmetrizatiosn: \n", affinity_matrix)

