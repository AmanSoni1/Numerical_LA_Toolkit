"""
Gaussian elimination without partial pivoting.

This implementation is intended only for numerical experiments.
"""

import numpy as np

from numerical_linear_alg.base import LinearSolver


class GaussianEliminationNoPivoting(LinearSolver):
    """Solve Ax = b using Gaussian elimination without pivoting."""

    name = "Gaussian elimination (no pivoting)"

    def solve(self, A, b):
        A = np.asarray(A, dtype=float).copy()
        b = np.asarray(b, dtype=float).copy()

        n = len(b)

        if A.shape != (n, n):
            raise ValueError("A must be a square matrix compatible with b.")

        self._forward_elimination(A, b)
        return self._back_substitution(A, b)

    @staticmethod
    def _forward_elimination(A, b):
        n = len(b)

        for k in range(n - 1):

            if A[k, k] == 0:
                raise np.linalg.LinAlgError("Zero pivot encountered.")

            for i in range(k + 1, n):

                multiplier = A[i, k] / A[k, k]

                A[i, k:] -= multiplier * A[k, k:]
                b[i] -= multiplier * b[k]

    @staticmethod
    def _back_substitution(A, b):
        n = len(b)
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):

            if A[i, i] == 0:
                raise np.linalg.LinAlgError("Zero diagonal entry encountered.")

            x[i] = (
                b[i] - np.dot(A[i, i + 1:], x[i + 1:])
            ) / A[i, i]

        return x


# Backward-compatible functional API
def gaussian_solve_without_pivoting(A, b):
    return GaussianEliminationNoPivoting().solve(A, b)
