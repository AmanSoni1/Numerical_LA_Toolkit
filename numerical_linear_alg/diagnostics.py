"""
Numerical diagnostics: how good is a computed solution x_hat of A x = b?

All norms are 2-norms: the Euclidean length for vectors and the spectral
norm (largest singular value) for matrices.
"""
import numpy as np


def residual(A, x, b):
    """
    Computes the residual r = b - Ax.

    It measures how closely the computed solution satisfies
    the given linear system Ax = b.
    """
    # Convert inputs into NumPy arrays
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)

    # Check whether A is a square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    
    # Check whether x and b are one-dimensional vectors
    if x.ndim != 1 or b.ndim != 1:
        raise ValueError("Invalid vector dimensions")
    
    # Check whether dimensions are compatible
    if A.shape[1] != x.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible shapes")

    return b - A @ x


def relative_residual(A, x, b):
    """
    Computes the relative residual ||b - Ax||_2 / ||b||_2.
    Measures the residual relative to the magnitude of b.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    r = residual(A, x, b)

    denominator = np.linalg.norm(b, 2)

    if denominator == 0:
        raise ValueError("b = 0")
    
    # Relative residual = ||b - Ax|| / ||b||
    return np.linalg.norm(r, 2) / denominator


def backward_error(A, x, b):
    """
    Computes the normwise relative backward error:
    ||b - Ax||_2 / (||A||_2 ||x||_2 + ||b||_2).

    Measures the relative size of the perturbations needed
    to make the computed solution an exact solution.
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)

    r = residual(A, x, b)

    denominator = (
        np.linalg.norm(A, 2) * np.linalg.norm(x, 2)
        + np.linalg.norm(b, 2)
    )

    if denominator == 0:
        raise ValueError("Zero denominator")
    
    # Backward error
    return np.linalg.norm(r, 2) / denominator


def condition_number(A):
    """
    Computes the 2-norm condition number of matrix A.

    Measures the sensitivity of a linear system to small
    perturbations in its input data.
    """ 
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
        
    # Calculate the 2-norm condition number
    return np.linalg.cond(A, 2)
