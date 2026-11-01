import numpy as np


def lu_factor(A, tol=1e-14):
    A = np.array(A, dtype=float)

    # Check square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Check finite values
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix must contain finite values")

    n = A.shape[0]

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n):

        # Find pivot row
        p = k + np.argmax(np.abs(U[k:, k]))

        # Check pivot
        if abs(U[p, k]) < tol:
            raise ValueError("Singular matrix")

        # Swap rows
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]

        # Elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    b = np.asarray(b, dtype=float)
    n = L.shape[0]

    y = np.zeros_like(b, dtype=float)

    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    return y


def back_substitution_upper(U, y):
    n = U.shape[0]

    x = np.zeros_like(y, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - U[i, i+1:] @ x[i+1:]
        ) / U[i, i]

    return x

def lu_solve(P, L, U, b):
    b = np.asarray(b, dtype=float)

    # Check right-hand side size
    if b.ndim not in (1, 2) or b.shape[0] != U.shape[0]:
        raise ValueError("Wrong size of b")

    y = forward_substitution(L, P @ b)
    x = back_substitution_upper(U, y)

    return x
