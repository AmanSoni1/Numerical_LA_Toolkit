
import numpy as np


def lu_factor(A, tol=1e-14):
    A = np.array(A, dtype=float)
    n = len(A)

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n):

        # Find pivot row
        p = k + np.argmax(abs(U[k:, k]))

        # Check pivot
        if abs(U[p, k]) < tol:
            raise ValueError("Matrix is singular")

        # Swap rows
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]

            # Swap previous multipliers
            L[[k, p], :k] = L[[p, k], :k]

        # Elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    return y


def back_substitution_upper(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

def lu_solve(P, L, U, b):
    y = forward_substitution(L, P @ b)
    x = back_substitution_upper(U, y)
    return x