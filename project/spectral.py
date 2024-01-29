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
    first_k_eigen_vectors = compute_first_k_eigen_vectors(A, k)
    # TODO: Apply K-means clustering on the selected eigenvectors

    k_for_k_mean = 2
    centroids, clusters = k_means_clustering(np.transpose(first_k_eigen_vectors), k_for_k_mean)
    # TODO: Return cluster labels

    return centroids, clusters


def compute_first_k_eigen_vectors(A, k):
    evalues, evectors = LA.eig(A)
    sorted_evalues = sorted(evalues)

    permutation_matrix = np.zeros((len(evalues), len(evalues)))
    for i in range(len(evalues)):
        item = evalues[i]

        index_in_sorted_array = sorted_evalues.index(item)
        permutation_matrix[i, index_in_sorted_array] = 1
    
    eigenvecto_matrices = evectors @ permutation_matrix


    return eigenvecto_matrices[:, :k]


# -------------------------- Quick Test -----------------------------
# A = np.array([[2, 3, 5],
#               [3, 4, 5],
#               [1, 2, 1]])

# scalers1 = np.array([2, 3, 1.5])
# scalers2 = np.array([[2], [3], [1.5]])
# scalers2 = scalers1.reshape(3, 1)
# scaled_matrix = A * scalers1 # scaled columns
# scaled_matrix = A * scalers2 # scaled rows
# print(scaled_matrix)

# -------------------------- Test Laplacian -----------------------------

# laplacian_A = laplacian(A)
# print("laplacian: \n", laplacian_A)

# ------------------------ compute the first k eigenvectors ------------------------

# A = np.array([[2, 3, 4],
#               [2, 4, 1],
#               [3, 6, 8]])

# A = np.array([[1, 2, 3], 
#               [2, 3, 4], 
#               [4, 5, 6]]) 

# P = np.array([[0, 1, 0],
#               [1, 0, 0],
#               [0, 0, 1]])

# evalues, evectors = LA.eig(A)
# sorted_evalues = sorted(evalues)

# print("evalues: \n", evalues)
# print("evectors: \n", evectors)
# print("sorted evalues: \n", sorted_evalues)

# permutation_matrix = np.zeros((len(evalues), len(evalues)))
# for i in range(len(evalues)):
#     item = evalues[i]

#     index_in_sorted_array = sorted_evalues.index(item)
#     permutation_matrix[i, index_in_sorted_array] = 1


# print(permutation_matrix)

# print("sorted_evectors:\n", evectors @ permutation_matrix)


# ----------------------- test compute_first_k_eigen_vectors -----------------------

# A = np.array([[1, 2, 3], 
#               [2, 3, 4], 
#               [4, 5, 6]]) 

# k = 2
# first_k_evectors = compute_first_k_eigen_vectors(A, k)
# print(first_k_evectors)

# ----------------------- produce data for k-mean -----------------------
# A = np.array([[2, 3, 5],
#               [3, 4, 5],
#               [1, 2, 1]])

# print(np.transpose(A))

# ----------------------- test spectral_clustering  -----------------------
A = np.array([[2, 3, 5],
              [3, 4, 5],
              [1, 2, 1]])

print(spectral_clustering(A, 2))