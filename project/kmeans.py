import numpy as np
# import matplotlib.pyplot as plt
# from numpy import linalg as LA

def k_means_clustering(data, k, max_iterations=100):    

    centroids, _ = create_centroid_cluster_structure(data, k)

    nearest_centroid_idx_for_each_point = None
    clusters = {}
    for iteration in range(max_iterations):
        distances_sq = np.sum((data[:, None, :] - centroids[None, :, :])**2, axis=-1)

        nearest_centroid_idx_for_each_point = np.argmin(distances_sq, axis=1) # Output: [0 1 2 2 0]
        clusters = {}

        unique_cluster_idx = np.unique(nearest_centroid_idx_for_each_point)

        for idx in unique_cluster_idx:
            cluster_data = data[nearest_centroid_idx_for_each_point == idx]
            clusters[idx] = cluster_data.tolist()  # Convert back to list for dictionary

        for cluster_key in range(len(clusters)):
            updated_centroid = np.mean(clusters.get(cluster_key), axis=0)
            centroids[cluster_key] = updated_centroid
    

    return nearest_centroid_idx_for_each_point, centroids 

def reset_clusters(clusters):
    for i in range(len(clusters)): # < k = 1 or 2 or 3
        clusters[i] = []

def create_centroid_cluster_structure(data, k):
    indices_array = np.arange(len(data))
    random_indices = np.random.choice(indices_array, size=k, replace=False)

    centroids = data[random_indices]
    clusters = dict()
    
    for i in range(k): # < 1 or 2 or 3 
        clusters[i] = []
    
    return centroids, clusters
            
        
