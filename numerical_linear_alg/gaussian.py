"""Gaussian elimination with partial pivoting.

Solves A x = b for a square matrix A by forward elimination with row swaps
(partial pivoting) followed by back substitution. NumPy is used only for
array storage and basic operations; the algorithm itself is implemented here.
"""

import numpy as np


def _as_system(A, b):
    """Validate A and b and return float copies (inputs are never modified)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    n = A.shape[0]
    if b.shape != (n,):
        raise ValueError(
            f"b must be a vector of length {n}, got shape {b.shape}"
        )
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("A and b must contain only finite values")
    return A, b


def back_substitution(U, y):
    """Solve U x = y for an upper-triangular matrix U.

    Raises
    ------
    ValueError
        If U is not square, shapes do not match, or a diagonal entry is zero.
    """
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"U must be a square matrix, got shape {U.shape}")
    n = U.shape[0]
    if y.shape != (n,):
        raise ValueError(f"y must be a vector of length {n}, got shape {y.shape}")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise ValueError(f"Zero on the diagonal of U at position {i}")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def gaussian_solve(A, b, tol=1e-14):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Coefficient matrix. Not modified.
    b : array_like, shape (n,)
        Right-hand side. Not modified.
    tol : float, optional
        A pivot is treated as unusable (matrix numerically singular) if
        ``abs(pivot) <= tol * max(abs(A))``.

    Returns
    -------
    x : ndarray, shape (n,)
        The computed solution.

    Raises
    ------
    ValueError
        If A is not square, b has the wrong shape, inputs contain NaN/inf,
        or a pivot is zero or negligibly small (singular or unusable A).
    """
    A, b = _as_system(A, b)
    n = A.shape[0]
    scale = np.max(np.abs(A)) if n > 0 else 0.0

    # Forward elimination with partial pivoting
    for k in range(n):
        # Row (at or below k) with the largest entry in column k
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if abs(A[p, k]) <= tol * scale:
            raise ValueError(
                f"Singular or unusable pivot in column {k}: "
                f"|pivot| = {abs(A[p, k]):.3e}"
            )
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]          # multiplier, |m| <= 1
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
            A[i, k] = 0.0                  # exact zero below the pivot

    return back_substitution(A, b)