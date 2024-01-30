import numpy as np
from kmeans import create_centroid_cluster_structure
from kmeans import k_means_clustering
import matplotlib.pyplot as plt

def generate_dataset(n, num_clusters, random_state=35):
    """
    Generates an array of n data points in 2D space, organized into specified clusters.

    Args:
        n (int): Number of data points to generate.
        num_clusters (int): Number of clusters to create within the dataset.
        random_state (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
        numpy.ndarray: Array of shape (n, 2) containing the generated data points.
    """

    np.random.seed(random_state)  # Set random seed for reproducibility

    # Create cluster centers with distinct coordinates
    centers = np.random.rand(num_clusters, 2) * 5  # Centers within [0, 5] range

    # Assign data points to clusters with a bit of noise
    data = np.vstack([center + np.random.randn(int(n/num_clusters), 2) * 0.4
                       for center in centers])

    return data



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

# ys, centroids = k_means_clustering(data, k)

# print("centroids: ", centroids)
# # print("clusters: ", clusters)

# for i in range(len(ys)):
#     if ys[i] == 0:
#         plt.plot(data[i, 0], data[i, 1], "r*")
#     elif ys[i] == 1:
#         plt.plot(data[i, 0], data[i, 1], "b*")
# plt.show()

# ---------------------------- test k_means_clustering ----------------------------
# data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [-1, -5], [3, -4], [-2, 4]], dtype=float)
# k = 2

# clusters, centroids = k_means_clustering(data, k)

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

# centers = [[2, 5], [8, 7], [4, 2]]  # Added a third cluster center
# stds = [0.5, 2, 1]  # Added a standard deviation for the third cluster
# data_1 = np.random.normal(loc=centers[0], scale=stds[0], size=(100, 2))
# data_2 = np.random.normal(loc=centers[1], scale=stds[1], size=(100, 2))
# data_3 = np.random.normal(loc=centers[2], scale=stds[2], size=(100, 2))  # Added data for the third cluster
# data = np.concatenate((data_1, data_2, data_3), axis=0)

# print("centroids: ", centroids)
# print("clusters: ", clusters)

# cluster_1 = np.array(clusters[0])
# cluster_2 = np.array(clusters[1])
# cluster_3 = np.array(clusters[2])

n = 300
num_clusters = 2
data = generate_dataset(n, num_clusters)
# print(data)


ys, centroids = k_means_clustering(data, num_clusters)

plt.plot(data[:, 0], data[:, 1], 'b*')

for i in range(len(ys)):
    if ys[i] == 0:
        plt.plot(data[i, 0], data[i, 1], "r*")
    elif ys[i] == 1:
        plt.plot(data[i, 0], data[i, 1], "b*")
    else:
        plt.plot(data[i, 0], data[i, 1], "y*")

plt.show()
    
# --------------------------------- using vectorization ----------------------------------

# datapoints = np.array([[1, 2], [3, 5], [4, 8], [10, 12], [3, -2]])
# centroids = np.array([[1, 2], [2, 5], [3, 8]])

# distances_sq = np.sum((datapoints[:, None, :] - centroids[None, :, :])**2, axis=-1)
# nearest_centroid_idx = np.argmin(distances_sq, axis=1)

# print(distances_sq)
# print(nearest_centroid_idx)

# distances_sq = np.sum((datapoints[:, None, :] - centroids[None, :, :])**2, axis=-1)
# print("distances_sq=\n", distances_sq)
# nearest_centroid_idx_for_each_point = np.argmin(distances_sq, axis=1) # Output: [0 1 2 2 0]
# print("nearest_centroid_idx_for_each_point=\n", nearest_centroid_idx_for_each_point)

# --------------------------------- test nearest_centroid_idx_for_each_point --------------------------------
# datapoints = np.array([[1, 2], [3, 5], [4, 8], [10, 12], [3, -2]])

# k = 2
# ys, centroids = k_means_clustering(datapoints, k)
# print("ys=\n", ys)

