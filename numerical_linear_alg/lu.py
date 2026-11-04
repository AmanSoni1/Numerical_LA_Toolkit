import numpy as np
from .base import LinearSolver


class LUDecomposition:

    def __init__(self, P, L=None, U=None, tol=1e-14):

        # If only A is given, calculate its LU decomposition
        if L is None and U is None:
            A = np.array(P, dtype=float)

            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("Matrix must be square")

            if not np.all(np.isfinite(A)):
                raise ValueError("Matrix must contain finite values")

            n = A.shape[0]
            U = A.copy()
            L = np.eye(n)
            P = np.eye(n)

            for k in range(n):
                p = k + np.argmax(np.abs(U[k:, k]))

                if abs(U[p, k]) < tol:
                    raise ValueError("Singular matrix")

                if p != k:
                    U[[k, p]] = U[[p, k]]
                    P[[k, p]] = P[[p, k]]
                    L[[k, p], :k] = L[[p, k], :k]

                for i in range(k + 1, n):
                    L[i, k] = U[i, k] / U[k, k]
                    U[i, k:] -= L[i, k] * U[k, k:]

        self.P = P
        self.L = L
        self.U = U

    # Construction
    def from_matrix(self, A, tol=1e-14):
        return LUDecomposition(A, tol=tol)

    # Access to factors
    def factors(self):
        return self.P, self.L, self.U

    # Solving
    def solve(self, b):
        b = np.asarray(b, dtype=float)

        if b.ndim not in (1, 2) or b.shape[0] != self.U.shape[0]:
            raise ValueError("Wrong size of b")

        y = self.forward_substitution(self.P @ b)
        return self.back_substitution(y)

    def forward_substitution(self, b):
        L = self.L
        b = np.asarray(b, dtype=float)
        n = L.shape[0]

        y = np.zeros_like(b, dtype=float)

        for i in range(n):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

        return y

    def back_substitution(self, y):
        U = self.U
        n = U.shape[0]

        x = np.zeros_like(y, dtype=float)

        for i in range(n - 1, -1, -1):
            x[i] = (
                y[i] - U[i, i+1:] @ x[i+1:]
            ) / U[i, i]

        return x

    def __repr__(self):
        return f"LUDecomposition(n={self.U.shape[0]})"


class LUSolver(LinearSolver):

    name = "LU factorization (partial pivoting)"

    def __init__(self, tol=1e-14):
        self.tol = tol

    def factorize(self, A):
        return LUDecomposition(A, tol=self.tol)

    def solve(self, A, b):
        return self.factorize(A).solve(b)


# Backward-compatible functional API
def lu_factor(A, tol=1e-14):
    obj = LUDecomposition(A, tol=tol)
    return obj.factors()


def lu_solve(P, L, U, b):
    return LUDecomposition(P, L, U).solve(b)
