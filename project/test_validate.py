import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA 
from validate import construct_affinity_matrix


# ------------------------- test construct_affinity_matrix -------------------------

# data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [-1, -5], [3, -4], [-2, 4]])
# affinity_type = 'knn'

# affinity_matrix = construct_affinity_matrix(data, affinity_type, k=4)
# print("affinity_matrix = \n", affinity_matrix)

# k_neighbors = 2
# distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=-1)
# print("distances: \n", distances)

# k_nearest_neighbors_indexes = np.argpartition(distances, k_neighbors, axis=1)[:, :k_neighbors]
# print("k_nearest_neighbors_indexes:\n", k_nearest_neighbors_indexes)

# ------------------------- test construct_affinity_matrix -------------------------
# from scipy.spatial.distance import cdist

# affinity_type = "knn"

# points = np.random.randint(0, 10, size=(10, 2))
# points = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [-1, -5], [3, -4], [-2, 4]])

# print("points:\n", points)

# Calculate pairwise distances
# distances = cdist(points, points)

# Get the indices of the 2 nearest neighbors (excluding itself)
# nearest_neighbors = np.argpartition(distances, 2, axis=1)[:, 1:3]

# Print the nearest neighbors for each point
# print("\nNearest neighbors for each point:")
# print(nearest_neighbors)


# affinity_matrix = construct_affinity_matrix(points, affinity_type)

# print("affinity_matrix:\n", affinity_matrix)
