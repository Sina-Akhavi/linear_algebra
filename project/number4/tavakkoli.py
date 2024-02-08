import numpy as np

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
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centered_A = A - centroid_A
    centered_B = B - centroid_B
    # TODO: Construct Cross-Covariance matrix
    H = np.dot(centered_A.T, centered_B)
    # TODO: Apply SVD to the Cross-Covariance matrix
    U, S, Vt = np.linalg.svd(H)
    # TODO: Calculate the rotation matrix
    R = np.dot(Vt.T, U.T)
    # Ensure a proper rotation matrix (handle reflections)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # TODO: Calculate the translation vector
    t = centroid_B - np.dot(R, centroid_A)

    # TODO: Return rotation and translation matrices
    return R, t