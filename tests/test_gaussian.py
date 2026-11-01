import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numerical_linear_alg.gaussian import gaussian_solve, back_substitution


# 1. Simple 2x2 system
def test_simple_system():
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([4, 7], dtype=float)

    x = gaussian_solve(A, b)

    assert np.allclose(x, [1, 2])


# 2. 3x3 system
def test_3x3_system():
    A = np.array([[4, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 4]], dtype=float)

    b = np.array([8, -10, 11], dtype=float)

    x = gaussian_solve(A, b)

    assert np.allclose(x, [1, -1, 2])


# 3. Row swapping (partial pivoting)
def test_row_swap():
    A = np.array([[0, 1],
                  [1, 1]], dtype=float)

    b = np.array([4, 7], dtype=float)

    x = gaussian_solve(A, b)

    assert np.allclose(x, [3, 4])


# 4. Very small pivot
def test_tiny_pivot():
    A = np.array([[1e-12, 1],
                  [1, 1]], dtype=float)

    b = np.array([1 + 1e-12, 2], dtype=float)

    x = gaussian_solve(A, b)

    assert np.allclose(x, [1, 1], atol=1e-10)


# 5. Singular matrix
def test_singular_matrix():
    A = np.array([[1, 2],
                  [2, 4]], dtype=float)

    b = np.array([1, 2], dtype=float)

    with pytest.raises(ValueError):
        gaussian_solve(A, b)


# 6. Zero matrix
def test_zero_matrix():
    A = np.zeros((3, 3))
    b = np.ones(3)

    with pytest.raises(ValueError):
        gaussian_solve(A, b)


# 7. Non-square matrix
def test_non_square():
    A = np.ones((2, 3))
    b = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_solve(A, b)


# 8. Mismatched dimensions
def test_mismatched_b():
    A = np.eye(3)
    b = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_solve(A, b)


# 9. NaN input
def test_nan_input():
    A = np.array([[np.nan, 1],
                  [1, 1]], dtype=float)

    b = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_solve(A, b)


# 10. Original inputs should not change
def test_inputs_not_modified():
    A = np.array([[0, 1],
                  [1, 1]], dtype=float)

    b = np.array([1, 2], dtype=float)

    A_original = A.copy()
    b_original = b.copy()

    gaussian_solve(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


# 11. List and integer input
def test_list_input():
    A = [[2, 1],
         [1, 3]]

    b = [4, 7]

    x = gaussian_solve(A, b)

    assert np.allclose(x, [1, 2])


# 12. Compare with NumPy solution
def test_numpy_comparison():
    A = np.array([[3, 1, 2],
                  [1, 4, 1],
                  [2, 1, 5]], dtype=float)

    b = np.array([10, 12, 16], dtype=float)

    x = gaussian_solve(A, b)
    x_numpy = np.linalg.solve(A, b)

    assert np.allclose(x, x_numpy)


# 13. Back substitution
def test_back_substitution():
    U = np.array([[2, 1, -1],
                  [0, 3, 2],
                  [0, 0, 4]], dtype=float)

    b = np.array([1, 12, 12], dtype=float)

    x = back_substitution(U, b)

    assert np.allclose(x, [1, 2, 3])


# 14. Zero diagonal in back substitution
def test_back_substitution_zero():
    U = np.array([[1, 2],
                  [0, 0]], dtype=float)

    b = np.array([1, 2], dtype=float)

    with pytest.raises(ValueError):
        back_substitution(U, b)