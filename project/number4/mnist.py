import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two point clouds.

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - dist: float, the Chamfer distance between the two point clouds.
    """
    dimensional1 = point_cloud1.shape[0]
    dimensional2 = point_cloud2.shape[0]

    # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2
    distances1 = np.linalg.norm(point_cloud1[:, np.newaxis, :] - point_cloud2[np.newaxis, :, :], axis=-1)
    distances1 = np.min(distances1, axis=1)
    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
    distances2 = np.linalg.norm(point_cloud2[:, np.newaxis, :] - point_cloud1[np.newaxis, :, :], axis=-1)
    distances2 = np.min(distances2, axis=1)
    # TODO: Return Chamfer distance, sum of the average distances in both directions

    term1 = np.sum(np.power(distances1, 2)) / dimensional1
    term2 = np.sum(np.power(distances2, 2)) / dimensional2
    
    return term1 + term2

def rigid_transform(A, B):
    """
    Find the rigid (translation + rotation) transformation between two sets of points.

    Parameters:
    - A: numpy array, mxn representing m points in an n-dimensional space.
    - B: numpy array, mxn representing m points in an n-dimensional space.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    """

    # TODO: Subtract centroids to center the point clouds A and B
    m = A.shape[0]
    centroidA = np.sum(A, axis=0) / m
    centroidB = np.sum(B, axis=0) / m

    muA = A - centroidA
    muB = B - centroidB

    # TODO: Construct Cross-Covariance matrix
    cross_conariance = A.T @ B

    # TODO: Apply SVD to the Cross-Covariance matrix
    u, s, vh = LA.svd(cross_conariance)
    # TODO: Calculate the rotation matrix
    rotation_matrix = vh.T @ u.T
    # TODO: Calculate the translation vector
    t = muB - rotation_matrix @ muA # cuases error
    # TODO: Return rotation and translation matrices

    return rotation_matrix, t

def icp(source, target, max_iterations=100, tolerance=1e-5):
    """
    Perform ICP (Iterative Closest Point) between two sets of points.

    Parameters:
    - source: numpy array, mxn representing m source points in an n-dimensional space.
    - target: numpy array, mxn representing m target points in an n-dimensional space.
    - max_iterations: int, maximum number of iterations for ICP.
    - tolerance: float, convergence threshold for ICP.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    - transformed_source: numpy array, mxn representing the transformed source points.
    """
        
    # TODO: Iterate until convergence
    for i in range(max_iterations):
    # TODO: Find the nearest neighbors of target in the source
        distances = np.linalg.norm(target[:, np.newaxis, :] - source[np.newaxis, :, :], axis=-1)
        argmin_output = np.argmin(distances, axis=1)
        nearest_neighbors = source[argmin_output]

    # TODO: Calculate rigid transformation
        rotation, translation_vector = rigid_transform(source, nearest_neighbors)
    # TODO: Apply transformation to source points
        source = rotation @ source + translation_vector
    # TODO: Calculate Chamfer distance
        distance = chamfer_distance(source, target)
    # TODO: Check for convergence
        if distance < tolerance:
            return source
    # TODO: Return the transformed source
    return source
    # pass

def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    # TODO: Iterate over point clouds to fill affinity matrix
    m = point_clouds.shape[0]
    affinity_matrix = np.array((m, m), dtype=float)
    i = 0
    for i in range(m):
        for j in range(i + 1, m):
            # TODO: For each pair of point clouds, register them with each other
            
            transformed_source = icp(point_clouds[i], point_clouds[j])

    # TODO: Calculate symmetric Chamfer distance between registered clouds
            distance = chamfer_distance(transformed_source, point_clouds[j])

            affinity_matrix[i][j] = distance
            affinity_matrix[j][i] = distance

    return affinity_matrix
    


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Plot Ach using its first 3 eigenvectors


