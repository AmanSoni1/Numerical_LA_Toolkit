import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerical_linear_alg.lu import lu_factor, lu_solve


def _is_permutation(P):
    return (
        np.all(P.sum(axis=0) == 1)
        and np.all(P.sum(axis=1) == 1)
    )


MATRICES = [
    np.array([[4., 3.], [6., 3.]]),
    np.array([[0., 1.], [1., 1.]]),
    np.array([[1e-12, 1.], [1., 1.]]),
    np.array([[2., 1., 1.], [4., 3., 3.], [8., 7., 9.]]),
    np.array([[0., 2., 1.], [1., 1., 1.], [3., 0., 2.]])
]


@pytest.mark.parametrize("A", MATRICES)
def test_reconstruction_PA_equals_LU(A):
    P, L, U = lu_factor(A)
    assert np.allclose(P @ A, L @ U)


def test_reconstruction_random():
    rng = np.random.default_rng(0)

    for n in (3, 6, 12, 25):
        A = rng.standard_normal((n, n))
        P, L, U = lu_factor(A)
        assert np.allclose(P @ A, L @ U)


@pytest.mark.parametrize("A", MATRICES)
def test_structure_of_factors(A):
    P, L, U = lu_factor(A)
    n = len(A)

    assert np.allclose(L, np.tril(L))
    assert np.allclose(np.diag(L), 1)
    assert np.allclose(U, np.triu(U))
    assert _is_permutation(P)
    assert np.max(np.abs(L)) <= 1.0 + 1e-15


def test_pivoting_swaps_rows_for_tiny_pivot():
    A = np.array([[1e-12, 1.], [1., 1.]])
    P, L, U = lu_factor(A)

    assert np.allclose(P, [[0., 1.], [1., 0.]])


def test_no_swap_when_diagonal_dominates():
    A = np.array([[5., 1.], [1., 3.]])
    P, L, U = lu_factor(A)

    assert np.allclose(P, np.eye(2))


def test_solve_accuracy():
    A = np.array([[2., 1., 1.],
                  [4., 3., 3.],
                  [8., 7., 9.]])

    x_exact = np.array([1., -2., 3.])
    P, L, U = lu_factor(A)

    b = A @ x_exact
    x = lu_solve(P, L, U, b)

    assert np.allclose(x, x_exact)


def test_factorization_reuse_many_vectors():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    P, L, U = lu_factor(A)

    for _ in range(5):
        x_exact = rng.standard_normal(5)
        b = A @ x_exact
        x = lu_solve(P, L, U, b)

        assert np.allclose(x, x_exact)


def test_multiple_right_hand_sides_as_matrix():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 4))
    X_exact = rng.standard_normal((4, 3))

    P, L, U = lu_factor(A)
    X = lu_solve(P, L, U, A @ X_exact)

    assert X.shape == (4, 3)
    assert np.allclose(X, X_exact)


def test_matches_numpy_on_random_systems():
    rng = np.random.default_rng(3)

    for n in (3, 8, 15):
        A = rng.standard_normal((n, n))
        b = rng.standard_normal(n)

        P, L, U = lu_factor(A)
        x = lu_solve(P, L, U, b)

        assert np.allclose(x, np.linalg.solve(A, b))


def test_singular_matrix_raises():
    A = np.array([[1., 2.], [2., 4.]])

    with pytest.raises(ValueError, match="Singular"):
        lu_factor(A)


def test_non_square_raises():
    A = np.ones((2, 3))

    with pytest.raises(ValueError, match="square"):
        lu_factor(A)


def test_nan_raises():
    A = np.array([[np.nan, 1.], [1., 1.]])

    with pytest.raises(ValueError, match="finite"):
        lu_factor(A)


def test_wrong_size_b_raises():
    P, L, U = lu_factor(np.eye(3))

    with pytest.raises(ValueError):
        lu_solve(P, L, U, np.ones(2))


def test_input_not_modified():
    A = np.array([[0., 1.], [1., 1.]])
    A0 = A.copy()

    lu_factor(A)

    assert np.array_equal(A, A0)


def test_list_input():
    A = [[4, 3], [6, 3]]
    P, L, U = lu_factor(A)

    A = np.array(A, dtype=float)
    assert np.allclose(P @ A, L @ U)
=======
def lu_test():
    raise NotImplementedError

