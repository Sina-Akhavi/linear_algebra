import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA 

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score

def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """
    n = data.shape[0]
    affinity_matrix = np.zeros((n, n))

    # TODO: Compute pairwise distances
    distances = np.linalg.norm(data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2)

    if affinity_type == 'knn':
        # TODO: Find k nearest neighbors for each point
        for i in range(len(data)):
            
            distances = LA.norm(data - data[i], axis=1) 
            k_neares_neighbors_indexes = np.argpartition(distances, k)[:k]

            affinity_matrix[i, k_neares_neighbors_indexes] = 1

        # TODO: Construct symmetric affinity matrix?
            affinity_matrix[k_neares_neighbors_indexes, i] = 1

        # TODO: Return affinity matrix
        return affinity_matrix

    elif affinity_type == 'rbf':
        # TODO: Apply RBF kernel

        for i in range(n):
            for j in range(n):
                affinity_matrix[i, j] = np.exp(-((LA.norm(data[i] - data[j])) ** 2) / (2 * (sigma **2)))
        

        # TODO: Return affinity matrix
        return affinity_matrix
        
    else:
        raise Exception("invalid affinity matrix type")

if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']
    
    figure, axes = plt.subplots(nrows=3, ncols=4)

    num_algorithms = 4
    i = 0
    for ds_name in datasets:
        X = np.load("./datasets/%s/data.npy" % ds_name)
        y = np.load("./datasets/%s/target.npy" % ds_name)
        
        n = len(np.unique(y)) # number of clusters
        k = 4
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)

        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)

        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))
        print("---------------------------------\n\n")

        Ys = [y_km, y_rbf, y_knn, y]
        labels = ["K-means", "RBF", "KNN", "Ground Truth"]

        for j in range(num_algorithms):

            sc = axes[i, j].scatter(X[:, 0], X[:, 1], c=Ys[j], cmap="viridis", edgecolors='k' ,linewidth=0.5, s=35, marker="o", label=labels[j])
            axes[i, j].set_title(f'{labels[j]} ({ds_name})', fontdict={'fontsize': 10, 'color': 'blue'})
            axes[i, j].legend(*sc.legend_elements(), title='clusters', fontsize=5, title_fontsize=8)
            axes[i, j].margins(0.2)

        i = i + 1
        
    # TODO: Show subplots
    plt.tight_layout()
    plt.show()
    
        