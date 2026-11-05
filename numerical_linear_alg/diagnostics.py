"""
Numerical diagnostics: how good is a computed solution x_hat of A x = b?

All norms are 2-norms: the Euclidean length for vectors and the spectral
norm (largest singular value) for matrices.

Class hierarchy
---------------
MatrixDiagnostics(A)               -> properties of the matrix alone
    SolutionDiagnostics(A, x, b)   -> accuracy of a computed solution
"""
import numpy as np


class MatrixDiagnostics:
    """Diagnostics that depend only on the (square) matrix A."""

    def __init__(self, A):
        A = np.asarray(A, dtype=float)

        # Check whether A is a square matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")

        self.A = A

    def condition_number(self):
        """
        Computes the 2-norm condition number of matrix A.

        Measures the sensitivity of a linear system to small
        perturbations in its input data.
        """
        return np.linalg.cond(self.A, 2)


class SolutionDiagnostics(MatrixDiagnostics):
    """Accuracy measures for a computed solution x of A x = b."""

    def __init__(self, A, x, b):
        super().__init__(A)

        x = np.asarray(x, dtype=float)
        b = np.asarray(b, dtype=float)

        # Check whether x and b are one-dimensional vectors
        if x.ndim != 1 or b.ndim != 1:
            raise ValueError("Invalid vector dimensions")

        # Check whether dimensions are compatible
        if self.A.shape[1] != x.shape[0] or self.A.shape[0] != b.shape[0]:
            raise ValueError("Incompatible shapes")

        self.x = x
        self.b = b

    def residual(self):
        """
        Computes the residual r = b - Ax.

        It measures how closely the computed solution satisfies
        the given linear system Ax = b.
        """
        return self.b - self.A @ self.x

    def relative_residual(self):
        """
        Computes the relative residual ||b - Ax||_2 / ||b||_2.
        Measures the residual relative to the magnitude of b.
        """
        r = self.residual()

        denominator = np.linalg.norm(self.b, 2)

        if denominator == 0:
            raise ValueError("b = 0")

        # Relative residual = ||b - Ax|| / ||b||
        return np.linalg.norm(r, 2) / denominator

    def backward_error(self):
        """
        Computes the normwise relative backward error:
        ||b - Ax||_2 / (||A||_2 ||x||_2 + ||b||_2).

        Measures the relative size of the perturbations needed
        to make the computed solution an exact solution.
        """
        r = self.residual()

        denominator = (
            np.linalg.norm(self.A, 2) * np.linalg.norm(self.x, 2)
            + np.linalg.norm(self.b, 2)
        )

        if denominator == 0:
            raise ValueError("Zero denominator")

        # Backward error
        return np.linalg.norm(r, 2) / denominator

def residual(A, x, b):
    return SolutionDiagnostics(A, x, b).residual()


def relative_residual(A, x, b):
    return SolutionDiagnostics(A, x, b).relative_residual()


def backward_error(A, x, b):
    return SolutionDiagnostics(A, x, b).backward_error()


def condition_number(A):
    return MatrixDiagnostics(A).condition_number()
