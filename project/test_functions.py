import numpy as np
from kmeans import create_centroid_cluster_structure
from kmeans import k_means_clustering
import matplotlib.pyplot as plt

# data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [-1, -5], [3, -4], [-2, 4]])
# k = 2


# centroids, clusters = create_centroid_cluster_structure(data, k)

# print("centroids: ", centroids)
# print("clusters: ", clusters)

# -------------------------------- test2 --------------------------------
# Define cluster centers and standard deviations
# cluster_centers = [[2, 5], [8, 7]]
# stds = [1, 2]

# # Generate data points for each cluster
# data_1 = np.random.normal(loc=cluster_centers[0], scale=stds[0], size=(50, 2))
# data_2 = np.random.normal(loc=cluster_centers[1], scale=stds[1], size=(50, 2))


# # Combine data points from both clusters
# data = np.concatenate((data_1, data_2), axis=0)
# k = 2

# centroids, clusters = k_means_clustering(data, k)

# print("centroids: ", centroids)
# # print("clusters: ", clusters)


# cluster_1 = np.array(clusters[0])
# cluster_2 = np.array(clusters[1])

# plt.plot(data[:, 0], data[:, 1], 'g*')
# plt.plot(cluster_1[:, 0], cluster_1[:, 1], 'r*')
# plt.plot(cluster_2[:, 0], cluster_2[:, 1], 'b*')
# plt.show()

# ---------------------------- test k_means_clustering ----------------------------
# data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [-1, -5], [3, -4], [-2, 4]], dtype=float)
# k = 2

# centroids, clusters = k_means_clustering(data, k)

# cluster_1 = np.array(clusters[0])
# cluster_2 = np.array(clusters[1])

# plt.plot(data[:, 0], data[:, 1], 'g*')
# plt.plot(cluster_1[:, 0], cluster_1[:, 1], 'r*')
# plt.plot(cluster_2[:, 0], cluster_2[:, 1], 'b*')
# plt.show()

# print(type(clusters[0]))

# print("centroids: ", centroids)
# print("clusters: ", clusters)

# --------------------------------- test k_means_clustering ----------------------------------

# Define cluster centers and standard deviations for K=3
centers = [[2, 5], [8, 7], [4, 2]]  # Added a third cluster center
stds = [1, 2, 0.5]  # Added a standard deviation for the third cluster

# Generate data points for each cluster
data_1 = np.random.normal(loc=centers[0], scale=stds[0], size=(50, 2))
data_2 = np.random.normal(loc=centers[1], scale=stds[1], size=(50, 2))
data_3 = np.random.normal(loc=centers[2], scale=stds[2], size=(50, 2))  # Added data for the third cluster

# Combine data points from all three clusters
data = np.concatenate((data_1, data_2, data_3), axis=0)

# Set the number of clusters to K=3
k = 3

centroids, clusters = k_means_clustering(data, k)

# print("centroids: ", centroids)
# print("clusters: ", clusters)

cluster_1 = np.array(clusters[0])
cluster_2 = np.array(clusters[1])
cluster_3 = np.array(clusters[2])

plt.plot(data[:, 0], data[:, 1], 'g*')
plt.plot(cluster_1[:, 0], cluster_1[:, 1], 'r*')
plt.plot(cluster_2[:, 0], cluster_2[:, 1], 'b*')
plt.plot(cluster_3[:, 0], cluster_3[:, 1], 'y*')
plt.show()

# --------------------------------- using vectorization ----------------------------------

# datapoints = np.array([[1, 2], [3, 5], [4, 8], [10, 12], [3, -2]])
# centroids = np.array([[1, 2], [2, 5], [3, 8]])

# distances_sq = np.sum((datapoints[:, None, :] - centroids[None, :, :])**2, axis=-1)
# nearest_centroid_idx = np.argmin(distances_sq, axis=1)

# print(distances_sq)
# print(nearest_centroid_idx)