import numpy as np
import matplotlib.pyplot as plt

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

    # TODO: Compute pairwise distances

    if affinity_type == 'knn':
        # TODO: Find k nearest neighbors for each point

        # TODO: Construct symmetric affinity matrix

        # TODO: Return affinity matrix

        pass
    elif affinity_type == 'rbf':
        # TODO: Apply RBF kernel

        # TODO: Return affinity matrix

        pass
    else:
        raise Exception("invalid affinity matrix type")


if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']

    # TODO: Create and configure plot

    for ds_name in datasets:
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset['data']     # feature points
        y = dataset['target']   # ground truth labels
        n = len(np.unique(y))   # number of clusters

        k = 3
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        # TODO: Create subplots

    # TODO: Show subplots
