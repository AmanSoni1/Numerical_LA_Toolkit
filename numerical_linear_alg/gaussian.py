"""Gaussian elimination with partial pivoting"""
import numpy as np

from .base import LinearSolver


class GaussianEliminationSolver(LinearSolver):

    name = "Gaussian elimination (partial pivoting)"

    def __init__(self, tol=1e-14):
        # Pivot magnitude below which the matrix is declared singular.
        self.tol = tol

    def solve(self, A, b):
        A, b = self._validate(A, b)
        self._forward_elimination(A, b)
        return self.back_substitution(A, b)

    def back_substitution(self, U, b):
        """Solve U x = b for an upper-triangular U."""
        n = len(b)
        x = [0] * n

        for i in range(n - 1, -1, -1):
            s = 0

            for j in range(i + 1, n):
                s += U[i][j] * x[j]

            if U[i][i] == 0:
                raise ValueError("Zero diagonal element")

            x[i] = (b[i] - s) / U[i][i]

        return x

    def _validate(self, A, b):
        """Check the inputs and return private float copies of A and b."""
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("Inputs must contain only finite values")

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")

        if b.shape != (A.shape[0],):
            raise ValueError("b must have the correct length")

        return A, b

    def _forward_elimination(self, A, b):
        """Reduce (A, b) in place to upper-triangular form."""
        n = len(b)

        for k in range(n):

            # Find the row with the largest pivot
            p = k + np.argmax(np.abs(A[k:, k]))

            # Swap rows if needed
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]

            if abs(A[k, k]) < self.tol:
                raise ValueError("Matrix is singular")

            # Eliminate entries below the pivot
            for i in range(k + 1, n):
                m = A[i, k] / A[k, k]

                A[i, :] = A[i, :] - m * A[k, :]
                b[i] = b[i] - m * b[k]


def back_substitution(U, b):
    return GaussianEliminationSolver().back_substitution(U, b)

def gaussian_solve(A, b):
    return GaussianEliminationSolver().solve(A, b)