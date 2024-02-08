from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np

def laplacian(A):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """

    m = A.shape[0]
    diag_degree_matrix = np.sum(A, axis=1)
    inverse_sqr_degree = np.sqrt(1 / diag_degree_matrix)

    row_scalers = inverse_sqr_degree.reshape(inverse_sqr_degree.size, 1)
    col_scalers = inverse_sqr_degree

    row_scaled_matrix = A * row_scalers
    row_col_scaled_matrix = row_scaled_matrix * col_scalers

    return np.eye(m) - row_col_scaled_matrix

def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    # TODO: Compute Laplacian matrix
    laplacian_affinity = laplacian(affinity)

    # TODO: Compute the first k eigenvectors of the Laplacian matrix

    # U, _, _ = LA.svd(laplacian_affinity, full_matrices=False, lapack_driver="gesvd")
    U, _, _ = LA.svd(laplacian_affinity, full_matrices=False)
    # LA.eig
    first_k_eigen_vectors = U[:, -k:]

    # TODO: Apply K-means clustering on the selected eigenvectors

    labels, centroids = k_means_clustering(first_k_eigen_vectors, k)
    # TODO: Return cluster labels

    return labels


